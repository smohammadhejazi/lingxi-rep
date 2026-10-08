"""Run Lingxi v1.5 over a list of instances and collect the predictions.

Each instance runs in its own process (scripts/run_instance.py), `--workers` at
a time. A rerun of the same `--run-id` skips completed instances and resumes the
others from v1.5's per-agent caches. Writes, under work/<benchmark>/runs/<run_id>/:

    run.json            configuration: models, dataset, code, sandbox, retrieval file
    predictions.jsonl   one row per instance, readable by both harnesses
    summary.json        status, timings and errors per instance

    uv run python scripts/run_batch.py --benchmark pro --instances instance_lists/pro_smoke_10.txt --workers 2
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def preflight(args, history_path: Path, ids: list[str]) -> None:
    import requests

    docker = subprocess.run(["docker", "info", "-f", "{{.Architecture}}"], capture_output=True, text=True)
    if docker.returncode != 0:
        raise SystemExit(f"Docker is not reachable: {docker.stderr.strip()}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set")
    from src.agent.tool_set.deepwiki_tool import DEEPWIKI_URL

    base = DEEPWIKI_URL.split("/chat/")[0]
    try:
        requests.get(base, timeout=10)
    except requests.RequestException as e:
        raise SystemExit(f"DeepWiki is not reachable at {base} ({e}); start it with scripts/start_deepwiki.sh")
    retrieved = {path.parent.name for path in
                 (common.benchmark_dir(args.benchmark) / "retrieval" / "advisor").glob("*/result.json")}
    missing = [i for i in ids if i not in retrieved]
    if missing and not args.allow_missing_retrieval:
        raise SystemExit(
            f"{len(missing)} instance(s) have no Lingxi Advisor result (run scripts/retrieve.py first), "
            f"e.g. {missing[:3]}. --allow-missing-retrieval runs them without knowledge."
        )
    if not history_path.exists():
        raise SystemExit(f"No history file at {history_path}; run scripts/retrieve.py first")


def run_metadata(args, history_path: Path, ids: list[str]) -> dict:
    from src.agent import benchmark, sandbox
    from src.agent.tool_set import dev_knowledge, deepwiki_tool
    import src.workflow.knowledge_tts_for_swebench_workflow as workflow

    return {
        "run_id": args.run_id,
        "created": common.now(),
        "system": "Lingxi v1.5 (knowledge_tts_for_swebench_workflow.py, commit db17799) with re-implemented retrieval",
        "dataset": benchmark.dataset_identity(),
        "instances": ids,
        "workflow": {"decoder_iterations": args.decoder_iterations, "mapper_iterations": args.mapper_iterations,
                     "solver_iterations": args.solver_iterations},
        "models": {
            "agents": workflow.AGENT_MODEL,
            "agent_settings": "temperature 1; thinking budget 1024, 3072 output tokens (aggregator: 4096 / 8192)",
            "text_editor_tool_type": workflow.TEXT_EDITOR_TOOL_TYPE,
            "knowledge": dev_knowledge.KNOWLEDGE_MODEL,
            "deepwiki": {"provider": deepwiki_tool.DEEPWIKI_PROVIDER, "model": deepwiki_tool.DEEPWIKI_MODEL,
                         "url": deepwiki_tool.DEEPWIKI_URL},
            "advisor_judge": os.environ.get("LINGXI_ADVISOR_JUDGE_MODEL", "claude-haiku-4-5-20251001"),
        },
        "retrieval": {"history_file": str(history_path), "sha256": common.file_sha256(history_path)},
        "sandbox": sandbox.describe(),
        "code": common.git_state(),
    }


def run_one(args, run_dir: Path, instance_id: str) -> dict:
    inst_dir = run_dir / "instances" / instance_id
    inst_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(common.REPO_ROOT / "scripts" / "run_instance.py"),
           "--benchmark", args.benchmark, "--instance-id", instance_id, "--run-dir", str(run_dir),
           "--decoder-iterations", str(args.decoder_iterations),
           "--mapper-iterations", str(args.mapper_iterations),
           "--solver-iterations", str(args.solver_iterations)]
    if args.keep_workspaces:
        cmd.append("--keep-workspaces")
    if args.remove_images:
        cmd.append("--remove-images")
    t0 = time.time()
    with open(inst_dir / "stdout.log", "a") as out:
        completed = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=common.REPO_ROOT)
    result_path = inst_dir / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {
        "instance_id": instance_id, "status": "failed", "error": f"run_instance exited {completed.returncode}"}
    print(f"{instance_id}: {result['status']} in {time.time() - t0:.0f}s"
          + (f" ({result.get('error')})" if result["status"] != "completed" else ""), flush=True)
    return result


def collect(run_dir: Path, run_id: str, ids: list[str]) -> dict:
    rows, summary = [], {"run_id": run_id, "instances": {}}
    for instance_id in ids:
        inst_dir = run_dir / "instances" / instance_id
        result_path = inst_dir / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {"status": "not_run"}
        patch_path = inst_dir / "prediction.patch"
        patch = patch_path.read_text() if result.get("status") == "completed" and patch_path.exists() else ""
        # `model_name_or_path`/`model_patch` for the SWE-bench harness,
        # `model`/`prediction` for SWE-bench_Pro-os (as swebench-pro-cli-runner writes them).
        rows.append({"instance_id": instance_id, "model_name_or_path": run_id, "model_patch": patch,
                     "model": run_id, "prediction": patch})
        summary["instances"][instance_id] = {
            "status": result.get("status"), "non_empty_patch": bool(patch.strip()),
            "knowledge_used": (result.get("knowledge") or {}).get("used"),
            "stages": result.get("stages"), "error": result.get("error"),
        }
    statuses = [v["status"] for v in summary["instances"].values()]
    summary["counts"] = {
        "instances": len(ids),
        "completed": statuses.count("completed"),
        "failed": statuses.count("failed"),
        "not_run": statuses.count("not_run"),
        "non_empty_patches": sum(v["non_empty_patch"] for v in summary["instances"].values()),
    }
    with open(run_dir / "predictions.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    common.write_json(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", required=True, choices=common.BENCHMARKS)
    parser.add_argument("--instances", required=True, help="file with one instance id per line")
    parser.add_argument("--run-id", default=None, help="default: lingxi-v15-<model>-<benchmark>-<timestamp>")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--decoder-iterations", type=int, default=3)
    parser.add_argument("--mapper-iterations", type=int, default=1)
    parser.add_argument("--solver-iterations", type=int, default=1)
    parser.add_argument("--keep-workspaces", action="store_true")
    parser.add_argument("--remove-images", action="store_true",
                        help="remove each instance's images after it runs (saves disk; evaluation pulls them again)")
    parser.add_argument("--retry-failed", action="store_true", help="also rerun instances that failed")
    parser.add_argument("--allow-missing-retrieval", action="store_true")
    args = parser.parse_args()

    common.configure(args.benchmark)
    common.check_deepwiki_path()
    from src.agent import benchmark, sandbox

    ids = common.read_instance_ids(args.instances)
    history_path = Path(os.environ["LINGXI_HISTORY_ISSUE_FILE"])
    if args.run_id is None:
        model = os.environ.get("LINGXI_AGENT_MODEL", "claude-haiku-4-5-20251001")
        short = model.replace("claude-", "").replace("-20251001", "").replace("-", "")
        args.run_id = f"lingxi-v15-{short}-{args.benchmark}-{time.strftime('%Y%m%dT%H%M%S')}"
    run_dir = common.benchmark_dir(args.benchmark) / "runs" / args.run_id
    preflight(args, history_path, ids)
    for instance_id in ids:  # unknown ids fail here, before anything runs
        benchmark.get_instance(instance_id)
    sandbox.ensure_internal_network()
    sandbox.ensure_host_server_python()
    if not (run_dir / "run.json").exists():
        common.write_json(run_dir / "run.json", run_metadata(args, history_path, ids))

    pending = []
    for instance_id in ids:
        result_path = run_dir / "instances" / instance_id / "result.json"
        status = json.loads(result_path.read_text()).get("status") if result_path.exists() else None
        if status == "completed" or (status == "failed" and not args.retry_failed):
            continue
        pending.append(instance_id)
    print(f"Run {args.run_id}: {len(ids)} instances, {len(pending)} to run, {args.workers} at a time")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda instance_id: run_one(args, run_dir, instance_id), pending))

    summary = collect(run_dir, args.run_id, ids)
    print(json.dumps(summary["counts"]))
    print(f"Predictions: {run_dir / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
