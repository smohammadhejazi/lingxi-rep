"""Shared setup for the SWE-bench reproduction scripts.

The v1.5 modules read their configuration from the environment when they are
imported, so every script calls `configure()` before importing anything from
`src`. Layout of the working directory (`LINGXI_WORK_DIR`, default `work/`):

    work/runtime/                      LINGXI_RUNTIME_DIR: datasets, container
                                       workspaces, DeepWiki repo copies, caches
    work/<benchmark>/retrieval/        Lingxi Advisor results and the v1.5-format
                                       history file built from them
    work/<benchmark>/knowledge/        v1.5 knowledge caches (analysis + summary
                                       per historical issue)
    work/<benchmark>/runs/<run_id>/    one run: run.json, predictions.jsonl,
                                       instances/<id>/ (v1.5 agent caches, logs)
"""

import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = Path(os.environ.get("LINGXI_WORK_DIR") or REPO_ROOT / "work").resolve()
BENCHMARKS = ("verified", "pro")

# DeepWiki-Open drops every file whose absolute path has a component equal to one
# of these names (DEFAULT_EXCLUDED_DIRS in api/config.py at a5f39e3, after its
# `strip("./")`; the match is case-sensitive), so none of them may appear in the
# path of the repository copies it indexes.
DEEPWIKI_EXCLUDED_NAMES = {
    "__pycache__", "_docs", "_site", "bin", "bower_components", "build", "bzr", "coverage", "dist",
    "docs", "eclipse", "env", "git", "hg", "idea", "jspm_packages", "log", "logs", "mypy_cache",
    "node_modules", "obj", "out", "pytest_cache", "ruff_cache", "settings", "site-docs", "svn",
    "target", "temp", "tmp", "venv", "virtualenv", "vs", "vscode",
}


def benchmark_dir(benchmark: str) -> Path:
    return WORK_DIR / benchmark


def history_file(benchmark: str) -> Path:
    return benchmark_dir(benchmark) / "retrieval" / "history_issues.jsonl"


def configure(benchmark: str) -> None:
    """Point the v1.5 modules at this benchmark and the working directory."""
    if benchmark not in BENCHMARKS:
        raise SystemExit(f"--benchmark must be one of {BENCHMARKS}")
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)
    except ImportError:
        pass
    os.environ["LINGXI_BENCHMARK"] = benchmark
    os.environ.setdefault("LINGXI_RUNTIME_DIR", str(WORK_DIR / "runtime"))
    os.environ.setdefault("LINGXI_KNOWLEDGE_CACHE_DIR", str(benchmark_dir(benchmark) / "knowledge"))
    os.environ.setdefault("LINGXI_HISTORY_ISSUE_FILE", str(history_file(benchmark)))
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        if os.environ.get(name, "").lower() == "true":
            print(f"Note: {name}=true sends every prompt and output to LangSmith.", file=sys.stderr)
    # v1.5 imports itself both as `src.agent...` and as `agent...`.
    for path in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    Path(os.environ["LINGXI_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["LINGXI_KNOWLEDGE_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)


def check_deepwiki_path() -> None:
    repo_map = Path(os.environ["LINGXI_RUNTIME_DIR"]).resolve() / "repo_map"
    clash = [part for part in repo_map.parts if part in DEEPWIKI_EXCLUDED_NAMES]
    if clash:
        raise SystemExit(
            f"DeepWiki would index nothing under {repo_map}: its path contains {clash}. "
            "Set LINGXI_WORK_DIR or LINGXI_RUNTIME_DIR to a path without those names."
        )


def read_instance_ids(path: str) -> list[str]:
    """One instance id per line; blank lines and '#' comments ignored."""
    ids = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{path} lists an instance more than once")
    return ids


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state() -> dict:
    def git(*args):
        result = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True)
        return result.stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
