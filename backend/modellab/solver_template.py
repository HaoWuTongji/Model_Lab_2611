"""Fixed executable template. Inputs are data, never Python expressions."""

import json
import math
import sys
import time
from pathlib import Path

try:
    import highspy
except ImportError:
    highspy = None


def finite(value):
    return float(value) if value is not None and math.isfinite(value) else None


def _solve_highs(ir, time_limit, mip_gap, threads, seed, presolve):
    if highspy is None:
        raise RuntimeError("当前 Python 环境未安装 highspy")
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    if threads and int(threads) > 0:
        h.setOptionValue("threads", int(threads))
    h.setOptionValue("random_seed", int(seed))
    if mip_gap is not None:
        h.setOptionValue("mip_rel_gap", float(mip_gap))
    if presolve in ("on", "off"):
        h.setOptionValue("presolve", presolve)
    infinity = highspy.kHighsInf
    positions = {v["id"]: i for i, v in enumerate(ir["variables"])}

    def ok(status):
        if status == highspy.HighsStatus.kError:
            raise RuntimeError("HiGHS rejected compiled input")

    for i, v in enumerate(ir["variables"]):
        ok(
            h.addVar(
                -infinity if v["lower"] is None else v["lower"],
                infinity if v["upper"] is None else v["upper"],
            )
        )
        if v["domain"] != "continuous":
            ok(h.changeColIntegrality(i, highspy.HighsVarType.kInteger))
    for t in ir["objective"]["terms"]:
        ok(h.changeColCost(positions[t["var_id"]], t["coef"]))
    ok(h.changeObjectiveOffset(ir["objective"]["constant"]))
    ok(
        h.changeObjectiveSense(
            highspy.ObjSense.kMaximize
            if ir["objective"]["sense"] == "max"
            else highspy.ObjSense.kMinimize
        )
    )
    for r in ir["constraints"]:
        ok(
            h.addRow(
                -infinity if r["lower"] is None else r["lower"],
                infinity if r["upper"] is None else r["upper"],
                len(r["terms"]),
                [positions[t["var_id"]] for t in r["terms"]],
                [t["coef"] for t in r["terms"]],
            )
        )
    ok(h.run())
    status = h.getModelStatus()
    mapping = {
        "kOptimal": "optimal",
        "kInfeasible": "infeasible",
        "kUnbounded": "unbounded",
        "kUnboundedOrInfeasible": "infeasible_or_unbounded",
        "kTimeLimit": "time_limit",
        "kIterationLimit": "iteration_limit",
        "kInterrupt": "cancelled",
        "kHighsInterrupt": "cancelled",
        "kSolveError": "numerical_error",
        "kModelError": "input_error",
        "kLoadError": "input_error",
        "kPresolveError": "numerical_error",
        "kPostsolveError": "numerical_error",
    }
    termination = mapping.get(status.name, "unknown")
    info, solution = h.getInfo(), h.getSolution()
    incumbent = bool(
        solution.value_valid
        and info.primal_solution_status
        == highspy.SolutionStatus.kSolutionStatusFeasible
    )
    if termination in (
        "infeasible",
        "unbounded",
        "infeasible_or_unbounded",
        "input_error",
    ):
        incumbent = False
    is_mip = any(v["domain"] != "continuous" for v in ir["variables"])
    return dict(
        termination=termination,
        raw_status=h.modelStatusToString(status),
        solver_version=f"HiGHS {h.version()}",
        has_incumbent=incumbent,
        proven_optimal=termination == "optimal" and incumbent,
        objective_value=finite(h.getObjectiveValue()) if incumbent else None,
        best_bound=finite(info.mip_dual_bound)
        if is_mip
        else (finite(h.getObjectiveValue()) if termination == "optimal" else None),
        relative_gap=finite(info.mip_gap) if is_mip and incumbent else None,
        variables={
            v["id"]: float(solution.col_value[i]) for i, v in enumerate(ir["variables"])
        }
        if incumbent
        else None,
        solver_time=h.getRunTime(),
    )


def _solve_gurobi(ir, time_limit, mip_gap, threads, seed, presolve):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError("当前 Python 环境未安装 gurobipy") from exc

    m = gp.Model("modellab")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = float(time_limit)
    m.Params.Threads = int(threads)
    m.Params.Seed = int(seed)
    if mip_gap is not None:
        m.Params.MIPGap = float(mip_gap)
    if presolve == "on":
        m.Params.Presolve = 2
    elif presolve == "off":
        m.Params.Presolve = 0

    gvars = {}
    for v in ir["variables"]:
        lb = -GRB.INFINITY if v["lower"] is None else float(v["lower"])
        ub = GRB.INFINITY if v["upper"] is None else float(v["upper"])
        vtype = (
            GRB.BINARY
            if v["domain"] == "binary"
            else GRB.INTEGER
            if v["domain"] == "integer"
            else GRB.CONTINUOUS
        )
        gvars[v["id"]] = m.addVar(lb=lb, ub=ub, vtype=vtype, name=v["id"])

    obj_expr = gp.LinExpr(float(ir["objective"]["constant"]))
    for t in ir["objective"]["terms"]:
        obj_expr.addTerms(float(t["coef"]), gvars[t["var_id"]])
    m.setObjective(
        obj_expr,
        GRB.MAXIMIZE if ir["objective"]["sense"] == "max" else GRB.MINIMIZE,
    )
    for r in ir["constraints"]:
        lhs = gp.LinExpr()
        for t in r["terms"]:
            lhs.addTerms(float(t["coef"]), gvars[t["var_id"]])
        lo, hi = r["lower"], r["upper"]
        if lo is not None and hi is not None and lo == hi:
            m.addLConstr(lhs, GRB.EQUAL, float(lo), name=r["id"])
        else:
            if lo is not None:
                m.addLConstr(lhs, GRB.GREATER_EQUAL, float(lo), name=f"{r['id']}_ge")
            if hi is not None:
                m.addLConstr(lhs, GRB.LESS_EQUAL, float(hi), name=f"{r['id']}_le")

    m.optimize()
    status_map = {
        GRB.OPTIMAL: "optimal",
        GRB.INFEASIBLE: "infeasible",
        GRB.UNBOUNDED: "unbounded",
        GRB.INF_OR_UNBD: "infeasible_or_unbounded",
        GRB.TIME_LIMIT: "time_limit",
        GRB.ITERATION_LIMIT: "iteration_limit",
        GRB.INTERRUPTED: "cancelled",
        GRB.NUMERIC: "numerical_error",
    }
    termination = status_map.get(m.Status, "unknown")
    incumbent = m.SolCount > 0 and termination not in (
        "infeasible",
        "unbounded",
        "infeasible_or_unbounded",
    )
    is_mip = any(v["domain"] != "continuous" for v in ir["variables"])
    return dict(
        termination=termination,
        raw_status=f"Gurobi status={m.Status}",
        solver_version=f"Gurobi {'.'.join(map(str, gp.gurobi.version()))}",
        has_incumbent=incumbent,
        proven_optimal=termination == "optimal" and incumbent,
        objective_value=finite(m.ObjVal) if incumbent else None,
        best_bound=finite(getattr(m, "ObjBound", None)) if is_mip else (finite(m.ObjVal) if termination == "optimal" else None),
        relative_gap=finite(getattr(m, "MIPGap", None)) if is_mip and incumbent else None,
        variables={v["id"]: float(gvars[v["id"]].X) for v in ir["variables"]}
        if incumbent
        else None,
        solver_time=float(m.Runtime),
    )


def _solve_cplex(ir, time_limit, mip_gap, threads, seed, presolve):
    try:
        import cplex
    except ImportError as exc:
        raise RuntimeError("当前 Python 环境未安装 cplex") from exc

    prob = cplex.Cplex()
    prob.set_log_stream(None)
    prob.set_error_stream(None)
    prob.set_warning_stream(None)
    prob.set_results_stream(None)
    prob.parameters.timelimit.set(float(time_limit))
    if threads and int(threads) > 0:
        prob.parameters.threads.set(int(threads))
    prob.parameters.randomseed.set(int(seed))
    if mip_gap is not None:
        prob.parameters.mip.tolerances.mipgap.set(float(mip_gap))
    if presolve == "off":
        prob.parameters.preprocessing.presolve.set(0)
    elif presolve == "on":
        prob.parameters.preprocessing.presolve.set(1)

    is_mip = any(v["domain"] != "continuous" for v in ir["variables"])
    names = [v["id"] for v in ir["variables"]]
    lb = [-cplex.infinity if v["lower"] is None else float(v["lower"]) for v in ir["variables"]]
    ub = [cplex.infinity if v["upper"] is None else float(v["upper"]) for v in ir["variables"]]
    obj_map = {t["var_id"]: float(t["coef"]) for t in ir["objective"]["terms"]}
    obj = [obj_map.get(v["id"], 0.0) for v in ir["variables"]]
    types = "".join(
        "B" if v["domain"] == "binary" else "I" if v["domain"] == "integer" else "C"
        for v in ir["variables"]
    )
    if is_mip:
        prob.variables.add(obj=obj, lb=lb, ub=ub, types=types, names=names)
    else:
        prob.variables.add(obj=obj, lb=lb, ub=ub, names=names)
    prob.objective.set_offset(float(ir["objective"]["constant"]))
    prob.objective.set_sense(
        prob.objective.sense.maximize
        if ir["objective"]["sense"] == "max"
        else prob.objective.sense.minimize
    )

    lin_exprs, senses, rhs, range_vals = [], [], [], []
    for r in ir["constraints"]:
        vars_in_row = [t["var_id"] for t in r["terms"]]
        coefs_in_row = [float(t["coef"]) for t in r["terms"]]
        lin_exprs.append([vars_in_row, coefs_in_row])
        lo, hi = r["lower"], r["upper"]
        if lo is not None and hi is not None:
            if lo == hi:
                senses.append("E")
                rhs.append(float(lo))
                range_vals.append(0.0)
            else:
                senses.append("R")
                rhs.append(float(lo))
                range_vals.append(float(hi - lo))
        elif lo is not None:
            senses.append("G")
            rhs.append(float(lo))
            range_vals.append(0.0)
        elif hi is not None:
            senses.append("L")
            rhs.append(float(hi))
            range_vals.append(0.0)
    if lin_exprs:
        prob.linear_constraints.add(
            lin_expr=lin_exprs, senses=senses, rhs=rhs, range_values=range_vals
        )

    t0 = time.monotonic()
    prob.solve()
    runtime = time.monotonic() - t0
    st = prob.solution.get_status()
    st_str = prob.solution.get_status_string()
    low_str = st_str.lower()
    if "optimal" in low_str and "infeasible" not in low_str:
        termination = "optimal"
    elif "infeasible or unbounded" in low_str:
        termination = "infeasible_or_unbounded"
    elif "infeasible" in low_str:
        termination = "infeasible"
    elif "unbounded" in low_str:
        termination = "unbounded"
    elif "time limit" in low_str:
        termination = "time_limit"
    else:
        termination = "unknown"

    incumbent = False
    vals = None
    obj_val = None
    try:
        if prob.solution.is_primal_feasible() and termination not in (
            "infeasible",
            "unbounded",
            "infeasible_or_unbounded",
        ):
            vals_list = prob.solution.get_values()
            vals = {v["id"]: float(vals_list[i]) for i, v in enumerate(ir["variables"])}
            obj_val = finite(prob.solution.get_objective_value())
            incumbent = True
    except Exception:
        incumbent = False

    best_bound = None
    rel_gap = None
    if is_mip:
        try:
            best_bound = finite(prob.solution.MIP.get_best_objective())
            if incumbent:
                rel_gap = finite(prob.solution.MIP.get_mip_relative_gap())
        except Exception:
            pass
    elif termination == "optimal":
        best_bound = obj_val

    return dict(
        termination=termination,
        raw_status=f"CPLEX ({st}): {st_str}",
        solver_version=f"CPLEX {prob.get_version()}",
        has_incumbent=incumbent,
        proven_optimal=termination == "optimal" and incumbent,
        objective_value=obj_val,
        best_bound=best_bound,
        relative_gap=rel_gap,
        variables=vals,
        solver_time=runtime,
    )


def solve(
    ir,
    time_limit=30,
    solver="highs",
    mip_gap=None,
    threads=1,
    seed=0,
    presolve="auto",
):
    start = time.monotonic()
    result = dict(
        termination="unknown",
        has_incumbent=False,
        proven_optimal=False,
        objective_value=None,
        best_bound=None,
        relative_gap=None,
        variables=None,
        wall_time=0,
        solver_time=0,
        raw_status="",
        solver_version=solver,
    )
    # Contradictions are mathematical infeasibility, not invalid input.
    conflicts = [
        v["source_id"]
        for v in ir["variables"]
        if v["lower"] is not None and v["upper"] is not None and v["lower"] > v["upper"]
    ]
    conflicts += [
        r["source_id"]
        for r in ir["constraints"]
        if not r["terms"]
        and (
            (r["lower"] is not None and r["lower"] > 0)
            or (r["upper"] is not None and r["upper"] < 0)
        )
    ]
    if conflicts:
        result.update(
            termination="infeasible",
            raw_status="deterministic_bound_contradiction",
            conflict_sources=conflicts,
        )
        return result
    if not ir["variables"]:
        result.update(
            termination="optimal",
            raw_status="constant_model",
            has_incumbent=True,
            proven_optimal=True,
            objective_value=ir["objective"]["constant"],
            variables={},
        )
        return result

    solver_name = (solver or "highs").strip().lower()
    if solver_name == "highs":
        out = _solve_highs(ir, time_limit, mip_gap, threads, seed, presolve)
    elif solver_name == "gurobi":
        out = _solve_gurobi(ir, time_limit, mip_gap, threads, seed, presolve)
    elif solver_name == "cplex":
        out = _solve_cplex(ir, time_limit, mip_gap, threads, seed, presolve)
    else:
        raise RuntimeError(f"不支持的求解器：{solver_name}")

    result.update(out)
    result["wall_time"] = time.monotonic() - start
    return result


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    ir = json.loads((folder / "model_ir.json").read_text(encoding="utf-8"))
    config_path = folder / "config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.exists()
        else {}
    )
    result = solve(
        ir,
        time_limit=config.get("time_limit", 30),
        solver=config.get("solver", "highs"),
        mip_gap=config.get("mip_gap"),
        threads=config.get("threads", 1),
        seed=config.get("seed", 0),
        presolve=config.get("presolve", "auto"),
    )
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else folder / "result.json"
    destination.write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "termination": result["termination"],
                "objective_value": result["objective_value"],
            },
            ensure_ascii=False,
        )
    )
