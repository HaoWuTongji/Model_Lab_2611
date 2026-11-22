import argparse
import json
from pathlib import Path
from .compiler import compile_model, canonical
from .exporter import write_bundle
from .validation import validate_result


def main():
    parser = argparse.ArgumentParser(
        description="M1 local trusted CLI (not the online execution path)"
    )
    parser.add_argument("model")
    parser.add_argument("--data")
    parser.add_argument("--out", default=".runtime/cli")
    parser.add_argument("--solve", action="store_true")
    parser.add_argument("--time-limit", type=float, default=30)
    args = parser.parse_args()
    document = json.loads(Path(args.model).read_text())
    ir = compile_model(
        document, json.loads(Path(args.data).read_text()) if args.data else {}
    )
    write_bundle(ir, args.out, args.time_limit)
    if args.solve:
        from .solver_template import solve

        result = solve(ir, args.time_limit)
        result["validation"] = validate_result(ir, result)
        Path(args.out, "result.json").write_text(canonical(result))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("已编译至 " + args.out)


if __name__ == "__main__":
    main()
