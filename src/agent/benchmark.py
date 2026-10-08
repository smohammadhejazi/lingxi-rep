"""
Benchmark selection for the SWE-bench workflow.

v1.5 ran on SWE-bench Verified only: the dataset, the instance images and the
repository path (`/testbed`) were written into the code. This module keeps that
behaviour as the default and adds SWE-bench Pro, selected with
`LINGXI_BENCHMARK=pro`.

* verified: `SWE-bench/SWE-bench_Verified` at the revision swebench-pro-cli-runner
  pins. Its problem statements, base commits and patches are identical to
  `princeton-nlp/SWE-bench_Verified`, which v1.5 loaded; it adds the columns the
  swebench 5 harness scores with. Images `swebench/sweb.eval.x86_64.*`,
  repository at `/testbed`.
* pro: `ScaleAI/SWE-bench_Pro`, config `v1` (731 tasks), images
  `jefzda/sweap-images:*`, repository at `/app`. On 2026-09-22 the dataset's
  default config became V2 (642 tasks, new images); V1 is kept byte-identical as
  config `v1`, which is what the swebench-pro-cli-runner baselines ran on.
"""

import json
import os
import re

from src.agent.constant import RUNTIME_DIR

VERIFIED = "verified"
PRO = "pro"

BENCHMARK = os.environ.get("LINGXI_BENCHMARK", VERIFIED).strip().lower()
if BENCHMARK not in (VERIFIED, PRO):
    raise ValueError(f"LINGXI_BENCHMARK must be '{VERIFIED}' or '{PRO}', got '{BENCHMARK}'")

VERIFIED_DATASET = os.environ.get("LINGXI_VERIFIED_DATASET", "SWE-bench/SWE-bench_Verified")
VERIFIED_REVISION = os.environ.get("LINGXI_VERIFIED_REVISION", "78f471bf655a3137b2e8a75af1501690ec009ec3")

PRO_DATASET = os.environ.get("LINGXI_PRO_DATASET", "ScaleAI/SWE-bench_Pro")
PRO_CONFIG = os.environ.get("LINGXI_PRO_CONFIG", "v1")
# The commit that added the `v1` config (SWE-bench Pro V2 release, 2026-09-22).
PRO_REVISION = os.environ.get("LINGXI_PRO_REVISION", "2d52cb3df914a3fcf80c7f66738b3a88ae37fc50")
PRO_DOCKERHUB_USERNAME = os.environ.get("LINGXI_PRO_DOCKERHUB_USERNAME", "jefzda")

REPO_DIR = "/app" if BENCHMARK == PRO else "/testbed"

_ROWS = None


def dataset_identity() -> dict:
    """The dataset this process reads, for run metadata."""
    if BENCHMARK == PRO:
        return {"benchmark": PRO, "dataset": PRO_DATASET, "config": PRO_CONFIG, "revision": PRO_REVISION}
    return {"benchmark": VERIFIED, "dataset": VERIFIED_DATASET, "config": None, "revision": VERIFIED_REVISION}


def _load_rows() -> dict:
    global _ROWS
    if _ROWS is None:
        from datasets import load_dataset

        if BENCHMARK == PRO:
            rows = load_dataset(
                PRO_DATASET, PRO_CONFIG, split="test", revision=PRO_REVISION, cache_dir=RUNTIME_DIR
            )
        else:
            rows = load_dataset(
                VERIFIED_DATASET, split="test", revision=VERIFIED_REVISION or None, cache_dir=RUNTIME_DIR
            )
        _ROWS = {row["instance_id"]: dict(row) for row in rows}
    return _ROWS


def get_instance(instance_id: str) -> dict:
    """The dataset row of `instance_id`; raises ValueError for an unknown id."""
    rows = _load_rows()
    if instance_id not in rows:
        raise ValueError(f"Invalid SWE instance id: {instance_id}")
    return dict(rows[instance_id])


def normalize_text(value) -> str:
    """Unwrap dataset fields stored as JSON-quoted string literals with escaped
    newlines, returning plain multi-line text. Same rule as the
    swebench-pro-cli-runner fix-agent prompt (agents/fixer/prompt.py)."""
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if text.startswith('"') and text.endswith('"'):
        try:
            decoded = json.loads(text)
            if isinstance(decoded, str):
                text = decoded
        except json.JSONDecodeError:
            pass
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n")
    return text


def issue_text(instance: dict) -> str:
    """The task text the agents receive.

    Verified: the problem statement, as in v1.5. Pro: the problem statement plus
    the task's Requirements and Interface, the sections SWE-bench Pro gives every
    solver (and the swebench-pro-cli-runner prompt renders the same way)."""
    if BENCHMARK != PRO:
        return instance["problem_statement"]
    sections = [normalize_text(instance.get("problem_statement", ""))]
    sections.append("## Requirements\n" + normalize_text(instance.get("requirements", "")))
    sections.append("## Interface\n" + normalize_text(instance.get("interface", "")))
    return "\n\n".join(sections)


def pro_image_uri(instance_id: str, repo: str) -> str:
    """Docker Hub image of a SWE-bench Pro task. Port of
    `helper_code/image_uri.py:get_dockerhub_image_uri` in SWE-bench_Pro-os, so the
    agents run in the image the evaluation harness scores in."""
    repo_base, repo_name_only = repo.lower().split("/")
    hsh = instance_id.replace("instance_", "")
    if instance_id == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
        repo_name_only = "element-web"
    elif "element-hq" in repo.lower() and "element-web" in repo.lower():
        repo_name_only = "element"
        if hsh.endswith("-vnan"):
            hsh = hsh[:-5]
    elif hsh.endswith("-vnan"):
        hsh = hsh[:-5]
    tag = f"{repo_base}.{repo_name_only}-{hsh}"
    if len(tag) > 128:
        tag = tag[:128]
    return f"{PRO_DOCKERHUB_USERNAME}/sweap-images:{tag}"


def source_image(instance: dict) -> str:
    """The published image the instance is solved and evaluated in."""
    if BENCHMARK == PRO:
        return pro_image_uri(instance["instance_id"], instance["repo"])
    if instance.get("image"):
        return instance["image"]
    repo, name = instance["instance_id"].split("__")
    return f"swebench/sweb.eval.x86_64.{repo}_1776_{name}:latest"


def fix_commit(instance: dict) -> str | None:
    """The commit that fixed a SWE-bench Pro task: the one the evaluation checks
    the tests out of (last line of `before_repo_set_cmd`). None for Verified."""
    if BENCHMARK != PRO:
        return None
    last = instance.get("before_repo_set_cmd", "").strip().splitlines()[-1]
    match = re.match(r"git checkout ([0-9a-f]{40}) -- ", last.strip())
    return match.group(1) if match else None
