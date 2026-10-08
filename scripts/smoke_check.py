"""Check the environment before a run: tools, keys, every model endpoint the run
uses, DeepWiki, and (with --instance-id) one instance container end to end.
Each model check makes one small API call.

    uv run python scripts/smoke_check.py --benchmark pro --instance-id <id>
"""

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

results = []


def check(name):
    def wrap(fn):
        def run(*args):
            try:
                detail = fn(*args)
                results.append((name, True, detail or ""))
                print(f"PASS  {name}  {detail or ''}", flush=True)
            except Exception as e:  # report and continue with the next check
                results.append((name, False, f"{type(e).__name__}: {e}"))
                print(f"FAIL  {name}  {type(e).__name__}: {e}", flush=True)
                if os.environ.get("SMOKE_TRACEBACK"):
                    traceback.print_exc()
        return run
    return wrap


@check("host tools")
def host_tools():
    missing = [t for t in ("rg", "docker", "git", "uv") if not shutil.which(t)]
    if missing:
        raise RuntimeError(f"missing {missing}")
    arch = subprocess.run(["docker", "info", "-f", "{{.Architecture}}"], capture_output=True, text=True).stdout.strip()
    if arch not in ("x86_64", "amd64"):
        raise RuntimeError(f"Docker runs on {arch}; the instance images are x86_64")
    return f"docker {arch}"


@check("API keys")
def keys():
    missing = [k for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GITHUB_TOKEN") if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"not set: {missing}")


@check("GitHub token")
def github():
    import requests

    r = requests.get("https://api.github.com/rate_limit",
                     headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"}, timeout=20)
    r.raise_for_status()
    res = r.json()["resources"]
    return f"core {res['core']['remaining']}/{res['core']['limit']}, search {res['search']['remaining']}/{res['search']['limit']}"


@check("agent model (v1.5 settings, tools incl. text editor)")
def agent_model():
    import src.workflow.knowledge_tts_for_swebench_workflow as w

    tools = [w.search_files_by_keywords, {"type": w.TEXT_EDITOR_TOOL_TYPE, "name": "str_replace_based_edit_tool"},
             w.ask_repository_agent, w.run_shell_cmd, w.think]
    reply = w.get_llm().bind_tools(tools).invoke(
        "Use the bash tool to run `ls` in the repository root. Do nothing else.")
    if not reply.tool_calls:
        raise RuntimeError(f"no tool call in reply: {str(reply.content)[:200]}")
    aggregator = w.get_llm(thinking_budget=4096, max_tokens=8192).invoke("Reply with the word ok.")
    return f"{w.AGENT_MODEL}: tool call {reply.tool_calls[0]['name']}; aggregator settings ok ({len(str(aggregator.content))} chars)"


@check("knowledge model")
def knowledge_model():
    from src.agent.tool_set import dev_knowledge

    reply = dev_knowledge.create_llm().invoke("Reply with the word ok.")
    return f"{dev_knowledge.KNOWLEDGE_MODEL}: {str(reply.content)[:40]!r}"


@check("Advisor relevance judge (Anthropic OpenAI-compatible endpoint)")
def advisor_judge():
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("LINGXI_ADVISOR_JUDGE_API_KEY") or os.environ["ANTHROPIC_API_KEY"],
                    base_url=os.environ.get("LINGXI_ADVISOR_JUDGE_BASE_URL", "https://api.anthropic.com/v1/"))
    model = os.environ.get("LINGXI_ADVISOR_JUDGE_MODEL", "claude-haiku-4-5-20251001")
    reply = client.chat.completions.create(model=model, temperature=0.0, max_tokens=64, messages=[
        {"role": "system", "content": "Answer with a JSON object."},
        {"role": "user", "content": 'Return {"ok": true}.'}])
    return f"{model}: {reply.choices[0].message.content[:40]!r}"


@check("DeepWiki (OpenAI embeddings + Claude answer)")
def deepwiki():
    from src.agent.constant import REPO_MAP_DIR
    from src.agent.tool_set.deepwiki_tool import ask_repository_agent

    name = "lingxi-smoke-repo"
    repo = Path(REPO_MAP_DIR) / name
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "calculator.py").write_text(
        "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n    return a + b\n\n\n"
        "def divide(a, b):\n    if b == 0:\n        raise ZeroDivisionError('b must not be zero')\n    return a / b\n")
    try:
        answer = ask_repository_agent.invoke({"query": "What does divide do when b is zero?"},
                                             config={"configurable": {"instance_id": name}})
    finally:
        shutil.rmtree(repo, ignore_errors=True)
    if not answer or answer.lstrip().startswith("Error") or "\nError" in answer[:300]:
        raise RuntimeError(str(answer)[:300])
    return f"{len(answer)} chars"


@check("instance container (prepared image, no network, tools, patch capture)")
def container(instance_id):
    import re
    from src.agent import benchmark, sandbox
    from src.agent.runtime_config import RuntimeConfig
    from src.agent.tool_set.edit_tool import str_replace_based_edit_tool
    from src.agent.tool_set.sepl_tools import run_shell_cmd, view_directory

    instance = benchmark.get_instance(instance_id)
    sandbox.prepare_image(instance)
    rc = RuntimeConfig()
    rc.load_from_swe_rex_docker_instance(instance_id)
    cfg = {"configurable": {"runtime_object": rc, "agent_name": "smoke", "instance_id": instance_id}}
    try:
        if not view_directory.invoke({"dir_path": "./", "depth": 0}, config=cfg):
            raise RuntimeError("empty directory listing")
        net = run_shell_cmd.invoke({"command": "timeout 10 git ls-remote https://github.com/git/git 2>&1 | tail -1"},
                                   config=cfg)
        if "Could not resolve host" not in net and "unable to access" not in net:
            raise RuntimeError(f"container reached the network: {net[:200]}")
        str_replace_based_edit_tool.invoke({"command": "create", "path": "lingxi_smoke.txt", "file_text": "ok\n"},
                                           config=cfg)
        patch = sandbox.capture_patch(rc)
        if "lingxi_smoke.txt" not in re.findall(r"(?m)^diff --git a/(\S+)", patch):
            raise RuntimeError("new file missing from the captured patch")
    finally:
        asyncio.run(rc.swe_rex_deployment.stop())
        RuntimeConfig.reset_instance()
        sandbox.cleanup_workspaces(instance)
    return f"{instance_id}: offline, tools and patch capture ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", default="pro", choices=common.BENCHMARKS)
    parser.add_argument("--instance-id", help="also start this instance's container")
    args = parser.parse_args()
    common.configure(args.benchmark)
    common.check_deepwiki_path()

    host_tools()
    keys()
    github()
    agent_model()
    knowledge_model()
    advisor_judge()
    deepwiki()
    if args.instance_id:
        container(args.instance_id)
    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed" + (f"; failed: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
