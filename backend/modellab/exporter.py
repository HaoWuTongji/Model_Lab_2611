from pathlib import Path
from .compiler import canonical, digest


def generate(ir):
    template = Path(__file__).with_name("solver_template.py").read_text(encoding="utf-8")
    # IDs are validated ASCII; generated comments are not user source text.
    comments = []
    mapping = {}
    for section, items in [
        ("variables", ir["variables"]),
        ("constraints", ir["constraints"]),
    ]:
        for position, item in enumerate(items):
            line = len(template.splitlines()) + len(comments) + 1
            comments.append(
                f"# source_id={item['source_id']} -> {section}[{position}] ({item['id']})"
            )
            mapping.setdefault(item["source_id"], []).append(
                dict(
                    ir_section=section,
                    ir_index=position,
                    instance_id=item["id"],
                    indices=item["indices"],
                    code_line=line,
                )
            )
    comments.append(f"# source_id={ir['objective']['source_id']} -> objective")
    mapping[ir["objective"]["source_id"]] = [
        dict(
            ir_section="objective", code_line=len(template.splitlines()) + len(comments)
        )
    ]
    return template + "\n".join(comments) + "\n", mapping


def _format_lin_terms(terms):
    if not terms:
        return "0"
    parts = []
    for idx, t in enumerate(terms):
        c = float(t["coef"])
        v = t["var_id"]
        if idx == 0:
            parts.append(f"{c:g} {v}")
        elif c >= 0:
            parts.append(f"+ {c:g} {v}")
        else:
            parts.append(f"- {abs(c):g} {v}")
    return " ".join(parts)


def generate_lp(ir):
    """Generate standard CPLEX/HiGHS/Gurobi-compatible .lp file deterministically from IR."""
    lines = [
        "\\ ModelLab standard LP export",
        "Maximize" if ir["objective"]["sense"] == "max" else "Minimize",
    ]
    obj_terms = _format_lin_terms(ir["objective"]["terms"])
    const = float(ir["objective"]["constant"])
    const_str = (
        f" + {const:g}" if const > 0 else f" - {abs(const):g}" if const < 0 else ""
    )
    lines.append(f" obj: {obj_terms}{const_str}")
    lines.append("Subject To")
    for r in ir["constraints"]:
        expr_str = _format_lin_terms(r["terms"])
        lo, hi = r["lower"], r["upper"]
        if lo is not None and hi is not None and lo == hi:
            lines.append(f" {r['id']}: {expr_str} = {lo:g}")
        else:
            if lo is not None:
                lines.append(f" {r['id']}_ge: {expr_str} >= {lo:g}")
            if hi is not None:
                lines.append(f" {r['id']}_le: {expr_str} <= {hi:g}")
    lines.append("Bounds")
    for v in ir["variables"]:
        lo, hi = v["lower"], v["upper"]
        if lo is None and hi is None:
            lines.append(f" {v['id']} free")
        elif lo is not None and hi is not None:
            lines.append(f" {lo:g} <= {v['id']} <= {hi:g}")
        elif lo is not None:
            lines.append(f" {v['id']} >= {lo:g}")
        else:
            lines.append(f" -inf <= {v['id']} <= {hi:g}")
    generals = [v["id"] for v in ir["variables"] if v["domain"] == "integer"]
    binaries = [v["id"] for v in ir["variables"] if v["domain"] == "binary"]
    if generals:
        lines.append("Generals")
        lines.append(" " + " ".join(generals))
    if binaries:
        lines.append("Binaries")
        lines.append(" " + " ".join(binaries))
    lines.append("End")
    return "\n".join(lines) + "\n"


def model_stats(ir):
    """Compute model size and numerical conditioning metrics."""
    n_cont = sum(1 for v in ir["variables"] if v["domain"] == "continuous")
    n_int = sum(1 for v in ir["variables"] if v["domain"] == "integer")
    n_bin = sum(1 for v in ir["variables"] if v["domain"] == "binary")
    coefs = [
        abs(float(t["coef"]))
        for t in ir["objective"]["terms"]
        if float(t["coef"]) != 0
    ]
    nnz = 0
    for r in ir["constraints"]:
        nnz += len(r["terms"])
        for t in r["terms"]:
            val = abs(float(t["coef"]))
            if val != 0:
                coefs.append(val)
    min_coef = min(coefs) if coefs else 0.0
    max_coef = max(coefs) if coefs else 0.0
    ratio = (max_coef / min_coef) if (min_coef > 0 and max_coef > 0) else 1.0
    return dict(
        variables=len(ir["variables"]),
        continuous=n_cont,
        integer=n_int,
        binary=n_bin,
        constraints=len(ir["constraints"]),
        nonzeros=nnz,
        min_coef=min_coef,
        max_coef=max_coef,
        condition_ratio=ratio,
    )


def write_bundle(
    ir,
    folder,
    time_limit=30,
    solver="highs",
    mip_gap=None,
    threads=1,
    seed=0,
    presolve="auto",
):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    code, mapping = generate(ir)
    lp_text = generate_lp(ir)
    config = dict(
        time_limit=time_limit,
        solver=solver,
        mip_gap=mip_gap,
        threads=threads,
        seed=seed,
        presolve=presolve,
    )
    for name, value in [
        ("model_ir.json", ir),
        ("source_map.json", mapping),
        ("config.json", config),
    ]:
        (folder / name).write_text(canonical(value), encoding="utf-8")
    (folder / "solve.py").write_text(code, encoding="utf-8")
    (folder / "model.lp").write_text(lp_text, encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    requirements = root / "runner" / "requirements.txt"
    if not requirements.exists():
        requirements = root / "requirements.txt"
    req_text = (
        requirements.read_text(encoding="utf-8")
        if requirements.exists()
        else "highspy>=1.10,<2\n"
    )
    (folder / "requirements.txt").write_text(req_text, encoding="utf-8")
    (folder / "README.txt").write_text(
        "Python 3.12: pip install -r requirements.txt; python solve.py\nAlso includes model.lp for standard LP/MILP solvers.\nInputs are data. No API credentials required.\n",
        encoding="utf-8",
    )
    return dict(
        ir_hash=digest(ir),
        code_hash=__import__("hashlib").sha256(code.encode()).hexdigest(),
    )
