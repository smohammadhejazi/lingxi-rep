"""Retrieve similar historical issues with Lingxi Advisor and write them as the
history file v1.5 reads.

v1.5 read `20250707-SwebenchCustom-WithHistoricIssue-Reranking-Summary-Filtered.jsonl`,
built offline with Qwen3-Embedding-8B and Qwen3-Reranker-4B; the authors could
not release it. This script runs the search & retrieval stage of Lingxi Advisor
(spine-se-lab/Lingxi-advisor, the authors' public reimplementation) with the
options of its batch preparation: live GitHub search, safety checks with the
temporal boundary, an LLM relevance gate, top 3. Only the retrieval stage runs;
the knowledge is written by v1.5's own code (scripts/run_instance.py).

    uv run python scripts/retrieve.py --benchmark pro --instances instance_lists/pro_smoke_10.txt
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# Lingxi Advisor's batch preparation defaults (capabilities/codify/batch_preparation/contracts.py).
SEARCH_CANDIDATE_COUNT = 5
FINAL_TOP_K = 3
JUDGE_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_OPENAI_BASE_URL = "https://api.anthropic.com/v1/"


def advisor_instance(instance: dict, benchmark_module) -> dict:
    """Only public task fields go to Advisor: never the gold patch or the tests."""
    public = {
        "repo": instance["repo"],
        "instance_id": instance["instance_id"],
        "base_commit": instance["base_commit"],
        "problem_statement": benchmark_module.normalize_text(instance["problem_statement"]),
    }
    if benchmark_module.BENCHMARK == benchmark_module.PRO:
        for field in ("requirements", "interface"):
            public[field] = benchmark_module.normalize_text(instance.get(field, ""))
        for field in ("repo_language", "issue_specificity", "issue_categories"):
            public[field] = instance.get(field) or ""
    return public


def advisor_env() -> dict:
    env = dict(os.environ)
    env.setdefault("LINGXI_ADVISOR_JUDGE_BASE_URL", ANTHROPIC_OPENAI_BASE_URL)
    env.setdefault("LINGXI_ADVISOR_JUDGE_MODEL", JUDGE_MODEL)
    if not env.get("LINGXI_ADVISOR_JUDGE_API_KEY"):
        env["LINGXI_ADVISOR_JUDGE_API_KEY"] = env.get("ANTHROPIC_API_KEY", "")
    if not env["LINGXI_ADVISOR_JUDGE_API_KEY"]:
        raise SystemExit("ANTHROPIC_API_KEY (or LINGXI_ADVISOR_JUDGE_API_KEY) is not set")
    if not env.get("GITHUB_TOKEN"):
        raise SystemExit("GITHUB_TOKEN is not set (Advisor searches GitHub)")
    return env


def history_rows(result: dict) -> list[dict]:
    """Advisor's selected candidates as rows of v1.5's history file."""
    rows = []
    for candidate in result.get("selected_candidates", []):
        commit = candidate.get("historical_fix_commit") or candidate.get("retrieved_commit_id") or ""
        position = candidate.get("final_selected_position")
        rows.append({
            "instance_id": candidate["instance_id"],
            "repo": candidate["repo"],
            "reranked_position": len(rows) if position is None else int(position),
            "relevance_score": float(candidate.get("llm_similarity_score") or 0.0),
            "retrieved_issue_number": int(candidate["historical_issue_number"]),
            "retrieved_issue_desc": candidate.get("historical_issue_description") or "",
            "retrieved_patch": candidate.get("historical_patch") or "",
            "retrieved_commit_id": commit,
            "historical_issue_url": candidate.get("historical_issue_url"),
            "advisor_misleading_risk": candidate.get("llm_misleading_risk"),
            "advisor_match_basis": candidate.get("llm_gate_match_basis"),
        })
    rows.sort(key=lambda row: row["reranked_position"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", required=True, choices=common.BENCHMARKS)
    parser.add_argument("--instances", required=True, help="file with one instance id per line")
    parser.add_argument("--advisor-bin", default=os.environ.get("LINGXI_ADVISOR_BIN")
                        or str(common.REPO_ROOT / "external" / "advisor-venv" / "bin" / "lingxi-advisor-candidate-search"))
    parser.add_argument("--refresh", action="store_true", help="re-run instances that already have a result")
    args = parser.parse_args()

    common.configure(args.benchmark)
    from src.agent import benchmark

    env = advisor_env()
    advisor_home = common.WORK_DIR / "advisor"
    out_root = common.benchmark_dir(args.benchmark) / "retrieval" / "advisor"
    ids = common.read_instance_ids(args.instances)
    failures = []
    for n, instance_id in enumerate(ids, 1):
        result_path = out_root / instance_id / "result.json"
        if result_path.exists() and not args.refresh:
            print(f"[{n}/{len(ids)}] {instance_id}: cached")
            continue
        instance_json = out_root / instance_id / "instance.json"
        common.write_json(instance_json, advisor_instance(benchmark.get_instance(instance_id), benchmark))
        cmd = [
            args.advisor_bin,
            "--instance-json", str(instance_json),
            "--project-root", str(advisor_home),
            "--cache-root", str(advisor_home / "cache"),
            "--output-root", str(advisor_home / "outputs"),
            "--repository-root", str(advisor_home / "repos"),
            "--search-candidate-count", str(SEARCH_CANDIDATE_COUNT),
            "--final-top-k", str(FINAL_TOP_K),
        ]
        print(f"[{n}/{len(ids)}] {instance_id}: searching", flush=True)
        completed = subprocess.run(cmd, capture_output=True, text=True, env=env)
        (out_root / instance_id / "advisor.log").write_text(completed.stderr)
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError:
            failures.append(instance_id)
            print(f"    no result (exit {completed.returncode}); see {out_root / instance_id / 'advisor.log'}")
            continue
        common.write_json(result_path, result)
        print(f"    {result.get('status')}: {result.get('selected_count')} selected", flush=True)
        if result.get("status") == "failed":
            failures.append(instance_id)

    # Rebuild the history file from every stored result, so it always matches them.
    rows = []
    for result_path in sorted(out_root.glob("*/result.json")):
        result = json.loads(result_path.read_text())
        if result.get("status") != "failed":
            rows.extend(history_rows(result))
    history = common.history_file(args.benchmark)
    history.parent.mkdir(parents=True, exist_ok=True)
    with open(str(history) + ".tmp", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    os.replace(str(history) + ".tmp", history)
    print(f"Wrote {len(rows)} rows for {len({r['instance_id'] for r in rows})} instances to {history}")
    if failures:
        print(f"Retrieval failed for {len(failures)} instance(s): {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
