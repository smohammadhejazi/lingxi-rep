"""Run Lingxi v1.5 on one instance, in its own process.

v1.5 keeps the runtime in a process-wide singleton, so each instance gets a
process (scripts/run_batch.py starts them). Stages:

1. prepare the instance image (src/agent/sandbox.py);
2. knowledge: for the instance's top-3 historical issues from the history file,
   build v1.5's analysis + summary with v1.5's own code (`contruct_history_knowledge`),
   before any agent starts, like the paper's offline knowledge base. A failed
   generation is retried; an issue whose knowledge still fails is dropped, and its
   decoder runs without knowledge, as v1.5 does when fewer than 3 issues exist;
3. DeepWiki: copy the repository for it and build its index (one question), so no
   agent waits for indexing;
4. the v1.5 workflow (`run_knowledge_workflow`): 3 decoders, aggregator, mapper,
   solver; every agent's output is cached in the instance directory, so a rerun
   resumes where it stopped;
5. cleanup of the repository copies left on the host.

    uv run python scripts/run_instance.py --benchmark pro --instance-id <id> --run-dir work/pro/runs/<run_id>
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

KNOWLEDGE_COUNT = 3
KNOWLEDGE_ATTEMPTS = 3
SUMMARY_TAGS = (
    "general_root_cause_analysis_steps", "bug_categorization", "relevant_architecture",
    "involved_components", "specific_involved_classes_functions_methods",
    "feature_or_functionality_of_issue", "general_fix_pattern", "summary_of_fix_checklist",
    "design_patterns_and_coding_practices", "additional_concepts",
)
DEEPWIKI_WARMUP_QUESTION = "What is the purpose of this repository and how is its source code organized?"


def knowledge_paths(dev_knowledge, row: dict) -> tuple[Path, Path]:
    """Cache files `contruct_history_knowledge` reads and writes for a row."""
    cache = Path(dev_knowledge.HISTORY_ISSUE_KNOWLEDGE_CACHE_DIR)
    stem = f"{row['repo'].replace('/', '+')}_{row['retrieved_issue_number']}"
    return cache / f"{stem}_step_analysis.txt", cache / f"{stem}_step_summary.txt"


def knowledge_ok(analysis: Path, summary: Path) -> bool:
    if not analysis.exists() or not summary.exists():
        return False
    analysis_text, summary_text = analysis.read_text(), summary.read_text()
    if analysis_text.startswith("Error generating") or summary_text.startswith("Error generating"):
        return False
    return any(f"<{tag}>" in summary_text for tag in SUMMARY_TAGS)


def ensure_knowledge(dev_knowledge, sandbox, instance_id: str, rows: list[dict], log) -> list[dict]:
    """Build (or reuse) the knowledge of each row; return the rows that have it."""
    from filelock import FileLock

    kept = []
    for row in rows:
        analysis, summary = knowledge_paths(dev_knowledge, row)
        with FileLock(os.path.join(sandbox.LOCK_DIR, f"knowledge-{analysis.stem}.lock")):
            for attempt in range(1, KNOWLEDGE_ATTEMPTS + 1):
                if knowledge_ok(analysis, summary):
                    break
                analysis.unlink(missing_ok=True)
                summary.unlink(missing_ok=True)
                log.info(f"Knowledge for {row['repo']}#{row['retrieved_issue_number']}: attempt {attempt}")
                dev_knowledge.contruct_history_knowledge(
                    project_name=row["repo"].replace("/", "+"),
                    issue_id=row["retrieved_issue_number"],
                    issue_description=row["retrieved_issue_desc"],
                    patch=row["retrieved_patch"],
                    patch_commit=row["retrieved_commit_id"],
                    instance_id=instance_id,
                )
                # The analysis runs in the singleton runtime; leave it unloaded.
                dev_knowledge.runtime_config.RuntimeConfig.reset_instance()
        if knowledge_ok(analysis, summary):
            kept.append(row)
        else:
            log.error(f"Knowledge for {row['repo']}#{row['retrieved_issue_number']} failed; dropping it")
    return kept


def warm_up_deepwiki(instance_id: str, log) -> str:
    from src.agent.constant import REPO_MAP_DIR
    from src.agent.runtime_config import RuntimeConfig
    from src.agent.tool_set.deepwiki_tool import ask_repository_agent

    if not os.path.isdir(os.path.join(REPO_MAP_DIR, instance_id)):
        rc = RuntimeConfig()
        rc.load_from_swe_rex_docker_instance(instance_id)  # makes the copy DeepWiki reads
        asyncio.run(rc.swe_rex_deployment.stop())
        RuntimeConfig.reset_instance()
    answer = ask_repository_agent.invoke(
        {"query": DEEPWIKI_WARMUP_QUESTION}, config={"configurable": {"instance_id": instance_id}}
    )
    if not answer or answer.lstrip().startswith("Error") or "\nError" in answer[:300]:
        raise RuntimeError(f"DeepWiki is not answering: {str(answer)[:500]}")
    log.info(f"DeepWiki ready ({len(answer)} chars)")
    return answer


def run_stages(args, instance: dict, inst_dir: Path, result: dict, log) -> None:
    from src.agent import sandbox
    from src.agent.tool_set import dev_knowledge
    import src.workflow.knowledge_tts_for_swebench_workflow as workflow

    instance_id = instance["instance_id"]
    t0 = time.time()
    result["prepared_image"] = sandbox.prepare_image(instance)
    result["stages"]["prepare_image_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    rows = [r for r in common.read_jsonl(Path(dev_knowledge.HISTORY_ISSUE_FILE)) if r["instance_id"] == instance_id]
    rows = sorted(rows, key=lambda r: r["reranked_position"])[:KNOWLEDGE_COUNT]
    kept = ensure_knowledge(dev_knowledge, sandbox, instance_id, rows, log)
    instance_history = inst_dir / "history_issues.jsonl"
    with open(instance_history, "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    # v1.5's knowledge extractor reads this module global at call time.
    dev_knowledge.HISTORY_ISSUE_FILE = str(instance_history)
    result["knowledge"] = {"retrieved": [r["retrieved_issue_number"] for r in rows],
                           "used": [r["retrieved_issue_number"] for r in kept]}
    result["stages"]["knowledge_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    warm_up_deepwiki(instance_id, log)
    result["stages"]["deepwiki_warmup_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    workflow.run_knowledge_workflow(
        instance_id,
        model_name=workflow.AGENT_MODEL,
        cache_dir=str(inst_dir),
        decoder_iterations=args.decoder_iterations,
        mapper_iterations=args.mapper_iterations,
        solver_iterations=args.solver_iterations,
    )
    result["stages"]["workflow_s"] = round(time.time() - t0, 1)
    patch_file = inst_dir / "problem_solver_0.patch"
    patch = patch_file.read_text() if patch_file.exists() else ""
    patch = patch.rstrip("\n") + "\n" if patch.strip() else ""
    (inst_dir / "prediction.patch").write_text(patch)
    result["patch_lines"] = len(patch.splitlines())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", required=True, choices=common.BENCHMARKS)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--decoder-iterations", type=int, default=3)
    parser.add_argument("--mapper-iterations", type=int, default=1)
    parser.add_argument("--solver-iterations", type=int, default=1)
    parser.add_argument("--keep-workspaces", action="store_true", help="keep the repository copies on the host")
    parser.add_argument("--remove-images", action="store_true", help="remove the prepared and source images afterwards")
    args = parser.parse_args()

    common.configure(args.benchmark)
    common.check_deepwiki_path()
    instance_id = args.instance_id
    inst_dir = Path(args.run_dir).resolve() / "instances" / instance_id
    inst_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(inst_dir)  # v1.5 writes logs/ relative to the working directory

    from langchain_core.callbacks import get_usage_metadata_callback
    from src.agent.logging_config import configure_logging
    from src.agent import benchmark, sandbox
    from src.agent.tool_set import dev_knowledge
    import src.workflow.knowledge_tts_for_swebench_workflow as workflow

    configure_logging(level=logging.INFO, log_dir=str(inst_dir / "log"), log_file=f"{instance_id}.log")
    log = logging.getLogger("lingxi.run_instance")

    result = {"instance_id": instance_id, "benchmark": args.benchmark, "started": common.now(),
              "agent_model": workflow.AGENT_MODEL, "knowledge_model": dev_knowledge.KNOWLEDGE_MODEL,
              "stages": {}, "status": "failed"}
    t_total = time.time()
    instance = benchmark.get_instance(instance_id)
    # Token usage of every model call made in this process (knowledge and agents);
    # DeepWiki's answers and Advisor's relevance checks run in other processes.
    with get_usage_metadata_callback() as usage:
        try:
            run_stages(args, instance, inst_dir, result, log)
            result["status"] = "completed"
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
            result["traceback"] = traceback.format_exc()
            log.error(f"{instance_id} failed: {result['error']}")
        finally:
            leaked = sandbox.remove_containers(instance)
            if leaked:
                log.warning(f"Removed {leaked} container(s) the run left running")
            result["containers_removed"] = leaked
            if not args.keep_workspaces:
                sandbox.cleanup_workspaces(instance)
            if args.remove_images:
                sandbox.remove_images(instance, source_too=True)
            result["usage"] = usage.usage_metadata
            result["finished"] = common.now()
            result["stages"]["total_s"] = round(time.time() - t_total, 1)
            common.write_json(inst_dir / "result.json", result)
    sys.exit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
