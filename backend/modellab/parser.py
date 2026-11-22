"""Restricted human expression notation using Python's syntax tree, never eval."""

import ast
from .compiler import ModelError


def parse_expression(text, model):
    if len(text) > 10000:
        raise ModelError("MODEL_LIMIT", "公式文本过长")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except (SyntaxError, RecursionError):
        raise ModelError(
            "PARSE_ERROR", "公式语法错误。示例：3*x + 5*y 或 sum(i, I, x[i])"
        )
    symbols = {v["id"]: "var" for v in model.get("variables", [])} | {
        p["id"]: "param" for p in model.get("parameters", [])
    }
    count = 0

    def visit(node, depth=0):
        nonlocal count
        count += 1
        if count > 20000 or depth > 30:
            raise ModelError("MODEL_LIMIT", "公式过深或过长")

        def child(n):
            return visit(n, depth + 1)

        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return dict(kind="number", value=node.value)
        if isinstance(node, ast.Name) and node.id in symbols:
            return dict(kind=symbols[node.id], id=node.id, indices=[])
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in symbols
        ):
            parts = (
                node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            )
            indices = []
            for part in parts:
                if isinstance(part, ast.Name):
                    indices.append(dict(kind="index", name=part.id))
                elif isinstance(part, ast.Constant) and type(part.value) in (str, int):
                    indices.append(dict(kind="literal", value=part.value))
                else:
                    raise ModelError(
                        "PARSE_ERROR", "索引须为索引名、整数或引号内的集合成员"
                    )
            return dict(kind=symbols[node.value.id], id=node.value.id, indices=indices)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return (
                dict(kind="neg", arg=child(node.operand))
                if isinstance(node.op, ast.USub)
                else child(node.operand)
            )
        if isinstance(node, ast.BinOp):
            a, b = child(node.left), child(node.right)
            if isinstance(node.op, ast.Add):
                return dict(kind="add", args=[a, b])
            if isinstance(node.op, ast.Sub):
                return dict(kind="add", args=[a, dict(kind="neg", arg=b)])
            if isinstance(node.op, ast.Mult):
                return dict(kind="mul", left=a, right=b)
            if isinstance(node.op, ast.Div):
                return dict(kind="div", numerator=a, denominator=b)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sum"
            and len(node.args) == 3
            and not node.keywords
            and isinstance(node.args[0], ast.Name)
        ):
            set_arg = node.args[1]
            if isinstance(set_arg, ast.Name):
                return dict(
                    kind="sum",
                    index=node.args[0].id,
                    set=set_arg.id,
                    body=child(node.args[2]),
                )
            if (
                isinstance(set_arg, ast.Subscript)
                and isinstance(set_arg.value, ast.Name)
                and not isinstance(set_arg.slice, ast.Tuple)
            ):
                sl = set_arg.slice
                if isinstance(sl, ast.Name):
                    set_index = dict(kind="index", name=sl.id)
                elif isinstance(sl, ast.Constant) and type(sl.value) in (str, int):
                    set_index = dict(kind="literal", value=sl.value)
                else:
                    raise ModelError(
                        "PARSE_ERROR", "集合索引须为索引名、整数或引号内的集合成员"
                    )
                return dict(
                    kind="sum",
                    index=node.args[0].id,
                    set=set_arg.value.id,
                    set_index=set_index,
                    body=child(node.args[2]),
                )
        raise ModelError(
            "PARSE_ERROR",
            "仅支持数值、已声明符号、加减乘除与 sum(i, I, 表达式)；不执行代码或任意 LaTeX",
        )

    return visit(tree.body)
