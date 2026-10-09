"""Retrieve similar historical issues with Lingxi Advisor and write them as the
history file v1.5 reads.

v1.5 read `20250707-SwebenchCustom-WithHistoricIssue-Reranking-Summary-Filtered.jsonl`,
built offline with Qwen3-Embedding-8B and Qwen3-Reranker-4B; the authors could
not release it. This script runs the search & retrieval stage of Lingxi Advisor
(spine-se-lab/Lingxi-advisor, the authors' public reimplementation) with the
options of its batch preparation (live GitHub search, safety checks with the
temporal boundary, an LLM relevance gate), except for the final selection. Only
the retrieval stage runs; the knowledge is written by v1.5's own code
(scripts/run_instance.py).

Selection follows v1.5's shape (top-N candidates, rerank, keep 3) rather than
Advisor's: 10 candidates instead of 5, and the gate's per-candidate scores rank
every safe candidate (Advisor's own order: similarity score, then misleading
risk, then search order) with no acceptance threshold. Advisor's thresholds kept
1 of 16 candidates on five Pro tasks; v1.5 always used the top 3 its reranker
returned. The gate's decision is kept with each row for analysis.

Instances with fewer than three safe candidates are searched again with
Advisor's "evaluation" strategy (scripts/advisor_evaluation_search.py): the
repository's full closed-issue catalog, every search phase, and fixes found
through commit messages. It costs far more GitHub calls, so it runs only for
them; v1.5's pool, the repository's whole history, almost always held three.

    uv run python scripts/retrieve.py --benchmark pro --instances instance_lists/pro_smoke_10.txt --workers 2
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# Advisor's batch default is 5 candidates (capabilities/codify/batch_preparation/contracts.py);
# v1.5 reranked an embedding top-20. 10 halves the GitHub calls of 20 (GraphQL fix discovery
# is the limit: 5,000 points/hour per account). Every candidate goes to the gate.
SEARCH_CANDIDATE_COUNT = 10
FINAL_TOP_K = 3
BOUNDED, EVALUATION = "bounded", "evaluation"
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


def _safe_candidates(result: dict) -> list[dict]:
    """Every candidate that passed Advisor's safety checks (temporal boundary,
    leakage) and has a patch, whatever the gate decided."""
    candidates = []
    for candidate in [*result.get("selected_candidates", []), *result.get("rejected_candidates", [])]:
        if candidate.get("post_retrieval_safety_passed") is False or candidate.get("leakage_decision") != "safe":
            continue
        if candidate.get("historical_issue_number") is None or not candidate.get("historical_patch"):
            continue
        candidates.append(candidate)
    return candidates


def history_rows(result: dict) -> list[dict]:
    """The top FINAL_TOP_K safe candidates as rows of v1.5's history file, ranked
    by the gate's scores without its threshold. Issues closed by the same fix are
    kept once."""
    def rank(candidate: dict) -> tuple:
        score, risk = candidate.get("llm_similarity_score"), candidate.get("llm_misleading_risk")
        order = candidate.get("candidate_position")
        return (-(score if score is not None else -1), risk if risk is not None else 99,
                order if order is not None else 10**6)

    rows, seen_fixes = [], set()
    for candidate in sorted(_safe_candidates(result), key=rank):
        commit = candidate.get("historical_fix_commit") or candidate.get("retrieved_commit_id") or ""
        fix = candidate.get("historical_patch_sha256") or commit or candidate["historical_patch"]
        if fix in seen_fixes:
            continue
        seen_fixes.add(fix)
        rows.append({
            "instance_id": candidate["instance_id"],
            "repo": candidate["repo"],
            "reranked_position": len(rows),
            "relevance_score": float(candidate.get("llm_similarity_score") or 0.0),
            "retrieved_issue_number": int(candidate["historical_issue_number"]),
            "retrieved_issue_desc": candidate.get("historical_issue_description") or "",
            "retrieved_patch": candidate.get("historical_patch") or "",
            "retrieved_commit_id": commit,
            "historical_issue_url": candidate.get("historical_issue_url"),
            "advisor_misleading_risk": candidate.get("llm_misleading_risk"),
            "advisor_match_basis": candidate.get("llm_gate_match_basis"),
            "advisor_gate_decision": candidate.get("llm_gate_decision"),
            "advisor_search_position": candidate.get("candidate_position"),
            "advisor_retrieval_strategy": result.get("lingxi_retrieval_strategy", BOUNDED),
        })
        if len(rows) == FINAL_TOP_K:
            break
    return rows


def retrieve_one(args, benchmark, env, out_root: Path, label: str, instance_id: str,
                 strategy: str = BOUNDED) -> bool:
    """Run Advisor for one instance; False when it produced no usable result.
    An evaluation-strategy rerun keeps the bounded result as result_bounded.json."""
    inst_dir = out_root / instance_id
    result_path = inst_dir / "result.json"
    advisor_home = common.WORK_DIR / "advisor"
    instance_json = inst_dir / "instance.json"
    common.write_json(instance_json, advisor_instance(benchmark.get_instance(instance_id), benchmark))
    if strategy == EVALUATION:
        entry = [str(Path(args.advisor_bin).parent / "python"), str(Path(__file__).parent / "advisor_evaluation_search.py")]
    else:
        entry = [args.advisor_bin]
    cmd = [
        *entry,
        "--instance-json", str(instance_json),
        "--project-root", str(advisor_home),
        "--cache-root", str(advisor_home / "cache"),
        "--output-root", str(advisor_home / "outputs"),
        "--repository-root", str(advisor_home / "repos"),
        "--search-candidate-count", str(SEARCH_CANDIDATE_COUNT),
        "--final-top-k", str(FINAL_TOP_K),
    ]
    print(f"{label} {instance_id}: searching ({strategy})", flush=True)
    completed = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log_name = "advisor.log" if strategy == BOUNDED else f"advisor_{strategy}.log"
    (inst_dir / log_name).write_text(completed.stderr)
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        print(f"{label} {instance_id}: no result (exit {completed.returncode}); see {inst_dir / log_name}", flush=True)
        return False
    result["lingxi_retrieval_strategy"] = strategy
    if strategy == EVALUATION and result_path.exists():
        os.replace(result_path, inst_dir / "result_bounded.json")
    common.write_json(result_path, result)
    print(f"{label} {instance_id}: {result.get('status')} ({strategy}): {len(_safe_candidates(result))} safe "
          f"candidates, {result.get('selected_count')} passed the gate, {len(history_rows(result))} kept", flush=True)
    return result.get("status") != "failed"


def needs_backfill(result_path: Path) -> bool:
    """A bounded result with fewer than FINAL_TOP_K usable candidates."""
    if not result_path.exists():
        return False
    result = json.loads(result_path.read_text())
    return (result.get("lingxi_retrieval_strategy", BOUNDED) == BOUNDED
            and result.get("status") != "failed" and len(history_rows(result)) < FINAL_TOP_K)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", required=True, choices=common.BENCHMARKS)
    parser.add_argument("--instances", required=True, help="file with one instance id per line")
    parser.add_argument("--advisor-bin", default=os.environ.get("LINGXI_ADVISOR_BIN")
                        or str(common.REPO_ROOT / "external" / "advisor-venv" / "bin" / "lingxi-advisor-candidate-search"))
    parser.add_argument("--refresh", action="store_true", help="re-run instances that already have a result")
    parser.add_argument("--workers", type=int, default=1,
                        help="Advisor processes at a time; each takes whole repositories, so no two "
                             "share a repository's cache. 2 suits one GitHub account")
    parser.add_argument("--no-backfill", action="store_true",
                        help="do not rerun instances left with fewer than 3 candidates in evaluation mode")
    args = parser.parse_args()

    common.configure(args.benchmark)
    from src.agent import benchmark

    env = advisor_env()
    out_root = common.benchmark_dir(args.benchmark) / "retrieval" / "advisor"
    ids = common.read_instance_ids(args.instances)
    todo = []
    for n, instance_id in enumerate(ids, 1):
        if (out_root / instance_id / "result.json").exists() and not args.refresh:
            print(f"[{n}/{len(ids)}] {instance_id}: cached")
        else:
            todo.append((f"[{n}/{len(ids)}]", instance_id))

    def run_grouped(items: list, strategy: str) -> list[str]:
        """Run `items` with `args.workers` workers, each taking whole repositories."""
        by_repo: dict[str, list] = {}
        for label, instance_id in items:
            by_repo.setdefault(benchmark.get_instance(instance_id)["repo"], []).append((label, instance_id))

        def run_repo(group: list) -> list[str]:
            return [instance_id for label, instance_id in group
                    if not retrieve_one(args, benchmark, env, out_root, label, instance_id, strategy)]

        failed = []
        # Largest repositories first, so the workers finish close together.
        groups = sorted(by_repo.values(), key=len, reverse=True)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for group_failed in pool.map(run_repo, groups):
                failed.extend(group_failed)
        return failed

    failures = run_grouped(todo, BOUNDED)

    # v1.5 always had three issues: its pool was the whole repository history.
    # Instances the bounded search left short are searched again exhaustively.
    short = [(f"[{n}/{len(ids)}]", instance_id) for n, instance_id in enumerate(ids, 1)
             if instance_id not in failures and needs_backfill(out_root / instance_id / "result.json")]
    if short and not args.no_backfill:
        print(f"Backfill: {len(short)} instance(s) with fewer than {FINAL_TOP_K} candidates, "
              f"rerun with Advisor's evaluation strategy", flush=True)
        failures += run_grouped(short, EVALUATION)

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
