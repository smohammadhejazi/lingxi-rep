"""Score a run's predictions with the official harness, called the way
swebench-pro-cli-runner calls it (src/swebench_runner/eval/{pro,verified}.py).

* pro: SWE-bench_Pro-os `swe_bench_pro_eval.py` with local Docker and the
  `jefzda` images, raw samples from the V1 dataset rows the run used;
* verified: `python -m swebench.harness.run_evaluation` (swebench 5) on the
  dataset the run used.

Both harnesses run from their own environments (scripts/setup_external.sh).
Writes work/<benchmark>/runs/<run_id>/evaluation/<timestamp>/ and prints the
resolved count.

    uv run python scripts/evaluate.py --benchmark pro --run-id <run_id> --workers 8
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

VERIFIED_TIMEOUT = 1800  # seconds per instance, swebench-pro-cli-runner's default


def evaluate_pro(predictions: list[dict], out_dir: Path, workers: int, redo: bool) -> dict:
    from src.agent import benchmark

    pro_os = Path(os.environ.get("LINGXI_PRO_OS_DIR") or common.REPO_ROOT / "external" / "SWE-bench_Pro-os")
    python = os.environ.get("LINGXI_PRO_OS_PYTHON") or str(common.REPO_ROOT / "external" / "pro-os-venv" / "bin" / "python")
    patches = [{"instance_id": p["instance_id"], "patch": p["prediction"], "prefix": p["model"]} for p in predictions]
    (out_dir / "patches.json").write_text(json.dumps(patches, indent=2))
    with open(out_dir / "raw_samples.jsonl", "w") as f:
        for p in predictions:
            f.write(json.dumps(benchmark.get_instance(p["instance_id"]), default=str) + "\n")
    cmd = [python, "swe_bench_pro_eval.py",
           "--raw_sample_path", str(out_dir / "raw_samples.jsonl"),
           "--patch_path", str(out_dir / "patches.json"),
           "--output_dir", str(out_dir),
           "--scripts_dir", str(pro_os / "run_scripts"),
           "--num_workers", str(workers),
           "--dockerhub_username", benchmark.PRO_DOCKERHUB_USERNAME,
           "--use_local_docker"]
    if redo:
        cmd.append("--redo")
    print(" ".join(cmd), flush=True)
    with open(out_dir / "harness.log", "w") as log:
        subprocess.run(cmd, cwd=pro_os, stdout=log, stderr=subprocess.STDOUT)
    results_path = out_dir / "eval_results.json"
    if not results_path.exists():
        raise SystemExit(f"The harness wrote no eval_results.json; see {out_dir / 'harness.log'}")
    results = json.loads(results_path.read_text())
    return {iid: bool(results.get(iid)) for iid in (p["instance_id"] for p in predictions)}


def evaluate_verified(predictions: list[dict], out_dir: Path, workers: int, run_id: str) -> dict:
    from src.agent import benchmark

    python = os.environ.get("LINGXI_SWEBENCH_PYTHON") or str(common.REPO_ROOT / "external" / "swebench-venv" / "bin" / "python")
    preds_path = out_dir / "predictions.jsonl"
    with open(preds_path, "w") as f:
        for p in predictions:
            f.write(json.dumps({"instance_id": p["instance_id"], "model_name_or_path": run_id,
                                "model_patch": p["model_patch"]}) + "\n")
    harness_run_id = f"{run_id}.{out_dir.name}"
    cmd = [python, "-m", "swebench.harness.run_evaluation",
           "--dataset_name", benchmark.VERIFIED_DATASET,
           "--split", "test",
           "--predictions_path", str(preds_path),
           "--max_workers", str(max(1, workers)),
           "--run_id", harness_run_id,
           "--timeout", str(VERIFIED_TIMEOUT),
           "--report_dir", str(out_dir)]
    print(" ".join(cmd), flush=True)
    with open(out_dir / "harness.log", "w") as log:
        subprocess.run(cmd, cwd=out_dir, stdout=log, stderr=subprocess.STDOUT)
    report_path = out_dir / f"{run_id}.{harness_run_id}.json"
    if not report_path.exists():
        raise SystemExit(f"The harness wrote no {report_path.name}; see {out_dir / 'harness.log'}")
    resolved = set(json.loads(report_path.read_text()).get("resolved_ids") or [])
    return {p["instance_id"]: p["instance_id"] in resolved for p in predictions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", required=True, choices=common.BENCHMARKS)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--redo", action="store_true", help="pro: re-run instances the harness already scored")
    args = parser.parse_args()

    common.configure(args.benchmark)
    run_dir = common.benchmark_dir(args.benchmark) / "runs" / args.run_id
    predictions = common.read_jsonl(run_dir / "predictions.jsonl")
    if not predictions:
        raise SystemExit(f"No predictions in {run_dir}; run scripts/run_batch.py first")
    run = json.loads((run_dir / "run.json").read_text())
    from src.agent import benchmark

    if run["dataset"] != benchmark.dataset_identity():
        raise SystemExit(f"The run used {run['dataset']}, this environment points at {benchmark.dataset_identity()}")
    out_dir = run_dir / "evaluation" / time.strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True)
    if args.benchmark == "pro":
        resolved = evaluate_pro(predictions, out_dir, args.workers, args.redo)
    else:
        resolved = evaluate_verified(predictions, out_dir, args.workers, args.run_id)
    summary = {
        "run_id": args.run_id,
        "evaluated": common.now(),
        "instances": len(resolved),
        "resolved": sum(resolved.values()),
        "resolve_rate": round(100 * sum(resolved.values()) / len(resolved), 2),
        "empty_patches": sum(1 for p in predictions if not p["prediction"].strip()),
        "per_instance": resolved,
    }
    common.write_json(out_dir / "summary.json", summary)
    print(f"Resolved {summary['resolved']}/{summary['instances']} ({summary['resolve_rate']}%); {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
