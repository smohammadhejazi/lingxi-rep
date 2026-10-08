"""Token usage and API cost of a run, from each instance's result.json.

Covers every model call made inside the instance processes: knowledge
generation and the six agents. DeepWiki's answers (its server) and Lingxi
Advisor's relevance checks (scripts/retrieve.py) run elsewhere; with a dedicated
Anthropic workspace, the Console's usage page has the complete total.

    uv run python scripts/cost_report.py --benchmark pro --run-id <run_id>
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# USD per million tokens: input, 5-minute cache write, cache read, output.
PRICES = {
    "claude-haiku-4-5-20251001": (1.00, 1.25, 0.10, 5.00),
}


def cost(model: str, usage: dict) -> float | None:
    if model not in PRICES:
        return None
    p_in, p_write, p_read, p_out = PRICES[model]
    details = usage.get("input_token_details") or {}
    read = details.get("cache_read", 0) or 0
    write = details.get("cache_creation", 0) or 0
    # langchain-anthropic counts cached tokens inside input_tokens.
    uncached = usage.get("input_tokens", 0) - read - write
    return (uncached * p_in + write * p_write + read * p_read + usage.get("output_tokens", 0) * p_out) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", required=True, choices=common.BENCHMARKS)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    run_dir = common.benchmark_dir(args.benchmark) / "runs" / args.run_id
    report = {"run_id": args.run_id, "instances": {}, "total_usd": 0.0, "tokens": {}}
    for result_path in sorted(run_dir.glob("instances/*/result.json")):
        result = json.loads(result_path.read_text())
        instance_usd = 0.0
        for model, usage in (result.get("usage") or {}).items():
            usd = cost(model, usage)
            if usd is None:
                print(f"No price for {model}; add it to PRICES", file=sys.stderr)
                continue
            instance_usd += usd
            totals = report["tokens"].setdefault(model, {"input": 0, "cache_read": 0, "cache_creation": 0, "output": 0})
            details = usage.get("input_token_details") or {}
            totals["input"] += usage.get("input_tokens", 0)
            totals["cache_read"] += details.get("cache_read", 0) or 0
            totals["cache_creation"] += details.get("cache_creation", 0) or 0
            totals["output"] += usage.get("output_tokens", 0)
        report["instances"][result["instance_id"]] = {
            "status": result.get("status"), "usd": round(instance_usd, 4),
            "minutes": round((result.get("stages") or {}).get("total_s", 0) / 60, 1),
        }
        report["total_usd"] += instance_usd
    n = len(report["instances"])
    report["total_usd"] = round(report["total_usd"], 2)
    report["mean_usd_per_instance"] = round(report["total_usd"] / n, 3) if n else None
    common.write_json(run_dir / "cost.json", report)
    for instance_id, row in report["instances"].items():
        print(f"{row['usd']:8.3f} USD  {row['minutes']:6.1f} min  {row['status']:10s} {instance_id}")
    print(f"Total {report['total_usd']} USD over {n} instances (mean {report['mean_usd_per_instance']})")
    print("Not included: DeepWiki answers and Lingxi Advisor relevance checks (see the Console).")


if __name__ == "__main__":
    main()
