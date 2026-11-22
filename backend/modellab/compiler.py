from __future__ import annotations
import ast
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from .schema import ModelDoc


class ModelError(Exception):
    def __init__(self, code, message, source_id=""):
        self.code, self.message, self.source_id = code, message, source_id
        super().__init__(message)

    def detail(self):
        return dict(code=self.code, message=self.message, source_id=self.source_id)


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass
class Limits:
    variables: int = 2000
    rows: int = 10000
    nonzeros: int = 200000
    nodes: int = 20000
    depth: int = 30
    magnitude: float = 1e12
    operations: int = 1000000


def guard_tree(raw, limits=Limits()):
    count = 0
    stack = [(raw, 0)]
    while stack:
        x, depth = stack.pop()
        count += 1
        if count > limits.nodes * 12 or depth > limits.depth * 3:
            raise ModelError("MODEL_LIMIT", "模型结构过大或过深")
        if isinstance(x, dict):
            stack.extend((v, depth + 1) for v in x.values())
        elif isinstance(x, list):
            stack.extend((v, depth + 1) for v in x)
        elif isinstance(x, float) and not math.isfinite(x):
            raise ModelError("INVALID_NUMBER", "拒绝 NaN/Infinity")


def compile_model(raw, data=None, revision_id="cli", limits=None):
    limits = limits or Limits()
    guard_tree(raw, limits)
    doc = ModelDoc.model_validate(raw).model_dump()
    data = data or {}
    guard_tree(data, limits)
    for issue in doc["unresolved_items"]:
        if not issue["resolved"] or not issue["resolution"].strip():
            raise ModelError("UNRESOLVED", issue["question"], issue["source_id"])
    seen = set()
    for s in [
        *doc["sets"],
        *doc["parameters"],
        *doc["variables"],
        doc["objective"],
        *doc["constraints"],
    ]:
        if s["id"] in seen:
            raise ModelError("DUPLICATE_ID", "标识符重复：" + s["id"], s["id"])
        seen.add(s["id"])
    sets = {}
    indexed_sets = {}
    set_index_of = {}
    for s in doc["sets"]:
        sid = s["id"]
        if s.get("index_set"):
            set_index_of[sid] = s["index_set"]
            mapping = {}
            union_members = []
            union_keys = set()
            for entry in s.get("indexed_members", []):
                ek = ((type(entry["index"]).__name__, entry["index"]),)
                if ek in mapping:
                    raise ModelError("DUPLICATE_MEMBER", "索引集合键重复", sid)
                sub_keys = set()
                for m in entry["members"]:
                    mk = (type(m).__name__, m)
                    if mk in sub_keys:
                        raise ModelError("DUPLICATE_MEMBER", "集合成员重复", sid)
                    sub_keys.add(mk)
                    if mk not in union_keys:
                        union_keys.add(mk)
                        union_members.append(m)
                mapping[ek] = entry["members"]
            indexed_sets[sid] = mapping
            sets[sid] = union_members
        else:
            members = s["members"]
            if len(set((type(v).__name__, v) for v in members)) != len(members):
                raise ModelError("DUPLICATE_MEMBER", "集合成员重复", sid)
            sets[sid] = members
    for sid, parent_set in set_index_of.items():
        if parent_set not in sets or parent_set in indexed_sets:
            raise ModelError("UNKNOWN_SET", "索引集合依赖了无效的父集合：" + parent_set, sid)
        parent_keys = {(type(v).__name__, v) for v in sets[parent_set]}
        for ek in indexed_sets[sid]:
            if ek[0] not in parent_keys:
                raise ModelError("INDEX_MEMBER", "索引集合键不在父集合中", sid)
    source = ""

    def fail(code, msg):
        raise ModelError(code, msg, source)

    def number(v):
        if (
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or abs(v) > limits.magnitude
        ):
            fail("INVALID_NUMBER", "数值非有限或超过允许范围")
        return float(v)

    def domains(names):
        for name in names:
            if name not in sets:
                fail("UNKNOWN_SET", "未知集合：" + name)
            if name in indexed_sets:
                fail("INDEX_SHAPE", "不能直接将索引集合作为变量或参数维度，请使用普通集合：" + name)
        return [sets[name] for name in names]

    def key(values):
        return tuple((type(v).__name__, v) for v in values)

    def is_subset(actual_set_id, expected_set_id):
        if actual_set_id == expected_set_id:
            return True
        actual_keys = {key([v]) for v in sets[actual_set_id]}
        expected_keys = {key([v]) for v in sets[expected_set_id]}
        return actual_keys.issubset(expected_keys)

    def validate_set_ref(set_id, set_index, scope):
        if set_id not in sets:
            fail("UNKNOWN_SET", "未知集合：" + set_id)
        if set_index is None:
            if set_id in indexed_sets:
                fail("INDEX_SHAPE", f"集合 {set_id} 是索引集合，请指定索引，如 {set_id}[j]")
            return
        if set_id not in indexed_sets:
            fail("INDEX_SHAPE", f"集合 {set_id} 不是索引集合，不能使用下标")
        parent_set = set_index_of[set_id]
        if set_index["kind"] == "index":
            if set_index["name"] not in scope:
                fail("INDEX_SCOPE", "索引未声明：" + set_index["name"])
            if not is_subset(scope[set_index["name"]], parent_set):
                fail("INDEX_SHAPE", "索引集合的下标与父集合不匹配")
        elif key([set_index["value"]]) not in {key([v]) for v in sets[parent_set]}:
            fail("INDEX_MEMBER", "索引集合下标不在父集合中")

    def resolve_set(set_id, set_index, env):
        if set_index is None:
            return sets[set_id]
        val = (
            env[set_index["name"]]
            if set_index["kind"] == "index"
            else set_index["value"]
        )
        return indexed_sets[set_id].get(key([val]), [])

    def combos(names):
        return itertools.product(*domains(names))

    params = {}
    declarations = {p["id"]: p for p in doc["parameters"]}
    for p in doc["parameters"]:
        source = p["id"]
        domains(p["sets"])
        value = (
            data.get(p["data_key"])
            if p["data_key"]
            else (p["entries"] if p["sets"] else p["value"])
        )
        if value is None and p["sets"] and p["default"] is not None:
            value = []
        if value is None:
            fail("MISSING_PARAMETER", "参数缺少数值：" + source)
        if not p["sets"]:
            params[source] = {(): number(value)}
            continue
        if not isinstance(value, list):
            fail("PARAMETER_SHAPE", "数组参数须使用 indices/value 条目")
        entries = {}
        for entry in value:
            if (
                not isinstance(entry, dict)
                or set(entry) != {"indices", "value"}
                or len(entry["indices"]) != len(p["sets"])
            ):
                fail("PARAMETER_SHAPE", "参数维度不匹配")
            k = key(entry["indices"])
            if k in entries:
                fail("PARAMETER_SHAPE", "参数索引重复")
            for index, s in zip(entry["indices"], p["sets"]):
                if key([index]) not in {key([v]) for v in sets[s]}:
                    fail("PARAMETER_SHAPE", "参数索引不在集合中")
            entries[k] = number(entry["value"])
        size = math.prod(len(sets[s]) for s in p["sets"])
        if p["default"] is None and len(entries) != size:
            fail("MISSING_PARAMETER", "参数数组不完整，缺失项不会默认为零")
        params[source] = entries
    variables, lookup = [], {}
    vdecl = {v["id"]: v for v in doc["variables"]}
    for v in doc["variables"]:
        source = v["id"]
        for indices in combos(v["sets"]):
            if len(variables) >= limits.variables:
                fail("MODEL_LIMIT", "展开变量超过上限")
            vid = f"v_{len(variables):06d}"
            lo = number(v["lower"]) if v["lower"] is not None else None
            hi = number(v["upper"]) if v["upper"] is not None else None
            if v["domain"] == "binary":
                lo, hi = (
                    max(0, lo) if lo is not None else 0,
                    min(1, hi) if hi is not None else 1,
                )
            variables.append(
                dict(
                    id=vid,
                    domain=v["domain"],
                    lower=lo,
                    upper=hi,
                    source_id=source,
                    indices=list(indices),
                    label=(v["symbol"] or source)
                    + (f"[{','.join(map(str, indices))}]" if indices else ""),
                )
            )
            lookup[(source, key(indices))] = vid
    operations = 0
    ast_nodes = 0

    def check(e, scope, depth=0):
        nonlocal ast_nodes
        ast_nodes += 1
        if ast_nodes > limits.nodes or depth > limits.depth:
            fail("MODEL_LIMIT", "表达式节点数或深度超限")
        kind = e["kind"]
        if kind == "number":
            number(e["value"])
            return False
        if kind in ("var", "param"):
            decls = vdecl if kind == "var" else declarations
            if e["id"] not in decls:
                fail("UNKNOWN_SYMBOL", "未知符号：" + e["id"])
            ds = decls[e["id"]]["sets"]
            if len(ds) != len(e["indices"]):
                fail("INDEX_SHAPE", "索引维度错误")
            for ix, sid in zip(e["indices"], ds):
                if ix["kind"] == "index":
                    if ix["name"] not in scope:
                        fail("INDEX_SCOPE", "索引未声明：" + ix["name"])
                    if not is_subset(scope[ix["name"]], sid):
                        fail("INDEX_SHAPE", "索引集合不匹配")
                elif key([ix["value"]]) not in {key([v]) for v in sets[sid]}:
                    fail("INDEX_MEMBER", "索引不在声明集合中")
            return kind == "var"
        if kind == "sum":
            validate_set_ref(e["set"], e.get("set_index"), scope)
            if e["index"] in scope:
                fail("INDEX_SCOPE", "禁止索引遮蔽，请使用不同索引")
            return check(e["body"], {**scope, e["index"]: e["set"]}, depth + 1)
        children = (
            e["args"]
            if kind == "add"
            else [e["arg"]]
            if kind == "neg"
            else [e["left"], e["right"]]
            if kind == "mul"
            else [e["numerator"], e["denominator"]]
        )
        flags = [check(c, scope, depth + 1) for c in children]
        if kind == "mul" and all(flags) or kind == "div" and flags[1]:
            fail("UNSUPPORTED_NONLINEAR", "不支持决策变量相乘或作分母")
        return any(flags)

    def add(a, b):
        c = number(a[0] + b[0])
        terms = a[1].copy()
        for k, v in b[1].items():
            terms[k] = number(terms.get(k, 0) + v)
            if terms[k] == 0:
                del terms[k]
        return c, terms

    def scale(a, k):
        return number(a[0] * k), {
            v: number(c * k) for v, c in a[1].items() if c * k != 0
        }

    def expr(e, env):
        nonlocal operations
        operations += 1
        if operations > limits.operations:
            fail("MODEL_LIMIT", "表达式展开计算量超限")
        kind = e["kind"]
        if kind == "number":
            return number(e["value"]), {}
        if kind in ("var", "param"):
            indices = [
                env[i["name"]] if i["kind"] == "index" else i["value"]
                for i in e["indices"]
            ]
            if kind == "var":
                return 0, {lookup[(e["id"], key(indices))]: 1}
            val = params[e["id"]].get(key(indices), declarations[e["id"]]["default"])
            if val is None:
                fail("MISSING_PARAMETER", "缺少参数值")
            return number(val), {}
        if kind == "neg":
            return scale(expr(e["arg"], env), -1)
        if kind == "add":
            out = (0, {})
            for arg in e["args"]:
                out = add(out, expr(arg, env))
            return out
        if kind == "sum":
            out = (0, {})
            for i in resolve_set(e["set"], e.get("set_index"), env):
                out = add(out, expr(e["body"], {**env, e["index"]: i}))
            return out
        if kind == "mul":
            a, b = expr(e["left"], env), expr(e["right"], env)
            return scale(a, b[0]) if not b[1] else scale(b, a[0])
        a, b = expr(e["numerator"], env), expr(e["denominator"], env)
        if b[0] == 0:
            fail("DIVISION_BY_ZERO", "参数分母为零")
        return scale(a, 1 / b[0])

    def terms(t):
        return [dict(var_id=k, coef=v) for k, v in sorted(t.items())]

    source = doc["objective"]["id"]
    check(doc["objective"]["expr"], {})
    constant, coef = expr(doc["objective"]["expr"], {})
    objective = dict(
        source_id=source,
        sense=doc["objective"]["sense"],
        constant=constant,
        terms=terms(coef),
    )
    rows = []
    nnz = len(coef)
    if nnz > limits.nonzeros:
        fail("MODEL_LIMIT", "目标非零系数超过上限")
    def compile_where(where_text, scope):
        text = (where_text or "").strip()
        if not text:
            return lambda _env: True
        if len(text) > 500:
            fail("MODEL_LIMIT", "适用范围条件过长")
        try:
            tree = ast.parse(text, mode="eval")
        except (SyntaxError, RecursionError):
            fail("PARSE_ERROR", "适用范围条件语法错误，示例：i != j 或 t > 1")

        def validate_node(node, depth=0):
            if depth > 20:
                fail("MODEL_LIMIT", "适用范围条件嵌套过深")
            if isinstance(node, ast.Constant):
                if type(node.value) not in (int, float, str, bool):
                    fail("PARSE_ERROR", "适用范围条件仅支持数字、字符串或布尔常量")
                return
            if isinstance(node, ast.Name):
                if node.id in scope:
                    return
                if node.id in declarations and not declarations[node.id]["sets"]:
                    return
                if node.id in vdecl:
                    fail("UNSUPPORTED_NONLINEAR", "适用范围条件不能引用决策变量：" + node.id)
                fail("UNKNOWN_SYMBOL", "适用范围条件引用了未定义符号：" + node.id)
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                pid = node.value.id
                if pid in vdecl:
                    fail("UNSUPPORTED_NONLINEAR", "适用范围条件不能引用决策变量：" + pid)
                if pid not in declarations:
                    fail("UNKNOWN_SYMBOL", "适用范围条件引用了未知参数：" + pid)
                parts = (
                    node.slice.elts
                    if isinstance(node.slice, ast.Tuple)
                    else [node.slice]
                )
                if len(parts) != len(declarations[pid]["sets"]):
                    fail("PARAMETER_SHAPE", "适用范围条件中的参数索引维度不匹配：" + pid)
                for part in parts:
                    if isinstance(part, ast.Name):
                        if part.id not in scope:
                            fail("UNKNOWN_INDEX", "适用范围条件索引不在量词范围内：" + part.id)
                    elif isinstance(part, ast.Constant) and type(part.value) in (str, int):
                        pass
                    else:
                        fail("PARSE_ERROR", "适用范围条件参数索引须为量词索引或常量")
                return
            if isinstance(node, ast.UnaryOp) and isinstance(
                node.op, (ast.Not, ast.USub, ast.UAdd)
            ):
                validate_node(node.operand, depth + 1)
                return
            if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
                for v in node.values:
                    validate_node(v, depth + 1)
                return
            if isinstance(node, ast.BinOp) and isinstance(
                node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
            ):
                validate_node(node.left, depth + 1)
                validate_node(node.right, depth + 1)
                return
            if isinstance(node, ast.Compare):
                validate_node(node.left, depth + 1)
                for op in node.ops:
                    if not isinstance(
                        op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
                    ):
                        fail("PARSE_ERROR", "适用范围条件仅支持 ==, !=, <, <=, >, >=")
                for comp in node.comparators:
                    validate_node(comp, depth + 1)
                return
            fail(
                "PARSE_ERROR",
                "适用范围条件仅支持索引/参数比较与 and/or/not 逻辑运算",
            )

        validate_node(tree.body)

        def eval_node(node, env):
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Name):
                if node.id in env:
                    return env[node.id]
                val = params[node.id].get((), declarations[node.id]["default"])
                if val is None:
                    fail("MISSING_PARAMETER", "缺少参数值")
                return number(val)
            if isinstance(node, ast.Subscript):
                pid = node.value.id
                parts = (
                    node.slice.elts
                    if isinstance(node.slice, ast.Tuple)
                    else [node.slice]
                )
                idx_vals = [
                    env[p.id] if isinstance(p, ast.Name) else p.value for p in parts
                ]
                val = params[pid].get(key(idx_vals), declarations[pid]["default"])
                if val is None:
                    fail("MISSING_PARAMETER", "缺少参数值")
                return number(val)
            if isinstance(node, ast.UnaryOp):
                v = eval_node(node.operand, env)
                if isinstance(node.op, ast.Not):
                    return not bool(v)
                if isinstance(node.op, ast.USub):
                    return -number(v)
                return number(v)
            if isinstance(node, ast.BoolOp):
                if isinstance(node.op, ast.And):
                    return all(bool(eval_node(v, env)) for v in node.values)
                return any(bool(eval_node(v, env)) for v in node.values)
            if isinstance(node, ast.BinOp):
                a, b = number(eval_node(node.left, env)), number(eval_node(node.right, env))
                if isinstance(node.op, ast.Add):
                    return a + b
                if isinstance(node.op, ast.Sub):
                    return a - b
                if isinstance(node.op, ast.Mult):
                    return a * b
                if b == 0:
                    fail("DIVISION_BY_ZERO", "适用范围条件中分母为零")
                return a / b
            if isinstance(node, ast.Compare):
                left = eval_node(node.left, env)
                for op, comp in zip(node.ops, node.comparators):
                    right = eval_node(comp, env)
                    try:
                        if isinstance(op, ast.Eq):
                            ok = type(left) is type(right) and left == right if isinstance(left, str) or isinstance(right, str) else left == right
                        elif isinstance(op, ast.NotEq):
                            ok = (type(left) is not type(right) or left != right) if isinstance(left, str) or isinstance(right, str) else left != right
                        elif isinstance(op, ast.Lt):
                            ok = left < right
                        elif isinstance(op, ast.LtE):
                            ok = left <= right
                        elif isinstance(op, ast.Gt):
                            ok = left > right
                        else:
                            ok = left >= right
                    except TypeError:
                        fail("PARSE_ERROR", "适用范围条件两侧类型不可比较")
                    if not ok:
                        return False
                    left = right
                return True
            return False

        return lambda env: bool(eval_node(tree.body, env))

    def expand_quantifiers(quantifiers):
        envs = [{}]
        for q in quantifiers:
            next_envs = []
            for env in envs:
                for val in resolve_set(q["set"], q.get("set_index"), env):
                    next_envs.append({**env, q["index"]: val})
            envs = next_envs
        return envs

    for c in doc["constraints"]:
        source = c["id"]
        scope = {}
        for q in c["for_each"]:
            if q["index"] in scope:
                fail("INDEX_SCOPE", "重复量词索引")
            validate_set_ref(q["set"], q.get("set_index"), scope)
            scope[q["index"]] = q["set"]
        check(c["lhs"], scope)
        check(c["rhs"], scope)
        where_fn = compile_where(c.get("where", ""), scope)
        for env in expand_quantifiers(c["for_each"]):
            if not where_fn(env):
                continue
            indices = [env[q["index"]] for q in c["for_each"]]
            if len(rows) >= limits.rows:
                fail("MODEL_LIMIT", "展开约束超过上限")
            k, t = add(expr(c["lhs"], env), scale(expr(c["rhs"], env), -1))
            nnz += len(t)
            if nnz > limits.nonzeros:
                fail("MODEL_LIMIT", "非零系数超过上限")
            rows.append(
                dict(
                    id=f"r_{len(rows):06d}",
                    source_id=source,
                    indices=list(indices),
                    lower=-k if c["sense"] in ("ge", "eq") else None,
                    upper=-k if c["sense"] in ("le", "eq") else None,
                    terms=terms(t),
                )
            )
    return dict(
        schema_version="1.0",
        model_revision_id=revision_id,
        compiler_version="0.1.0",
        objective=objective,
        variables=variables,
        constraints=rows,
    )
