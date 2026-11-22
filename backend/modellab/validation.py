"""Independent evaluator: no import from the solver or its executable template."""

import math


def validate_result(ir, result, atol=1e-6, rtol=1e-8, int_tol=1e-5):
    report = dict(
        passed=False,
        absolute_tolerance=atol,
        relative_tolerance=rtol,
        integer_tolerance=int_tol,
        rows=[],
        variables=[],
        objective=None,
    )
    if not result.get("has_incumbent"):
        report.update(passed=None, message="未取得可行解，不进行残差校验")
        return report
    values = result.get("variables")
    if (
        not isinstance(values, dict)
        or set(values) != {v["id"] for v in ir["variables"]}
        or any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            for x in values.values()
        )
    ):
        report["message"] = "变量集合不完整或包含非有限数值"
        return report

    def bounds(x, lo, hi, scale):
        residual = max(
            0, (lo - x) if lo is not None else 0, (x - hi) if hi is not None else 0
        )
        tolerance = atol + rtol * max(1, scale, abs(lo or 0), abs(hi or 0))
        return residual, tolerance

    passed = True
    for v in ir["variables"]:
        x = values[v["id"]]
        residual, tol = bounds(x, v["lower"], v["upper"], abs(x))
        integral = abs(x - round(x)) if v["domain"] != "continuous" else 0
        good = residual <= tol and integral <= int_tol
        report["variables"].append(
            dict(
                id=v["id"],
                source_id=v["source_id"],
                label=v.get("label", v["source_id"]),
                indices=v["indices"],
                value=x,
                violation=residual,
                integer_violation=integral,
                passed=good,
            )
        )
        passed &= good
    for r in ir["constraints"]:
        products = [t["coef"] * values[t["var_id"]] for t in r["terms"]]
        activity = math.fsum(products)
        residual, tol = bounds(
            activity, r["lower"], r["upper"], math.fsum(abs(x) for x in products)
        )
        good = math.isfinite(activity) and residual <= tol
        report["rows"].append(
            dict(
                id=r["id"],
                source_id=r["source_id"],
                indices=r["indices"],
                activity=activity,
                violation=residual,
                tolerance=tol,
                passed=good,
            )
        )
        passed &= good
    objective = ir["objective"]["constant"] + math.fsum(
        t["coef"] * values[t["var_id"]] for t in ir["objective"]["terms"]
    )
    supplied = result.get("objective_value")
    good = (
        isinstance(supplied, (float, int))
        and math.isfinite(supplied)
        and abs(supplied - objective) <= atol + rtol * max(1, abs(objective))
    )
    report["objective"] = dict(recomputed=objective, solver=supplied, passed=good)
    report["passed"] = bool(passed and good)
    return report
