"""Resolve a goal and optionally evaluate saved physics trace, without actuation."""

import argparse
import json
from pathlib import Path

from task_goal import resolve_goal
from task_success import SuccessEvaluator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--criteria", type=Path)
    p.add_argument("--trace", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if bool(args.trace) != bool(args.criteria):
        p.error("--trace and --criteria must be used together")

    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    catalog = read(args.catalog)
    goal = resolve_goal(read(args.request), catalog)
    result = {"goal": goal}
    if args.trace:
        evaluator = SuccessEvaluator(goal, catalog, read(args.criteria))
        with args.trace.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    evaluator.update(json.loads(line))
        result["evaluation"] = evaluator.finalize()
        result["trace"] = str(args.trace.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "target": goal["target_name"],
                "goal_id": goal["goal_id"],
                "status": result.get("evaluation", {}).get("status", "resolved"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
