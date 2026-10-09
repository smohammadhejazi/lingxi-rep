# Lingxi v1.5 on SWE-bench Pro with Claude Haiku 4.5

This branch (`v1.5-pro-haiku`) starts from commit `db17799`, the last commit with
v1.5's SWE-bench workflow (`src/workflow/knowledge_tts_for_swebench_workflow.py`;
the same file is in Zenodo record versions v1 and v2). It runs that workflow on
SWE-bench Pro (and SWE-bench Verified) with Claude Haiku 4.5. The workflow, its
prompts, tools and knowledge mechanics are v1.5's. The historical-issue retrieval
is reimplemented, because the authors could not release theirs.

## Running

Requirements: Linux x86_64 with Docker (the containers are reached on their
internal Docker address), `uv`, `git`, `unzip`, ripgrep (`rg`, used by v1.5's
search tool), and enough disk for the instance images (1–10 GB each).

```bash
cp .env.example .env              # ANTHROPIC_API_KEY, OPENAI_API_KEY, GITHUB_TOKEN
./scripts/setup_external.sh       # DeepWiki-Open, Lingxi Advisor, both harnesses, uv sync
./scripts/start_deepwiki.sh       # keep running (port 8008)
uv run python scripts/smoke_check.py --benchmark pro \
  --instance-id instance_qutebrowser__qutebrowser-0aa57e4f7243024fa4bba8853306691b5dbd77b3-v5149fcda2a9a6fe1d35dfed1bade1444a11ef271
```

Each experiment is the same four steps. `BENCH` is `verified` or `pro`, `LIST` an
instance list:

```bash
uv run python scripts/retrieve.py   --benchmark $BENCH --instances $LIST --workers 2   # Lingxi Advisor (GitHub + Haiku)
uv run python scripts/run_batch.py  --benchmark $BENCH --instances $LIST --workers 2 --run-id <id>
uv run python scripts/evaluate.py   --benchmark $BENCH --run-id <id> --workers 8
uv run python scripts/cost_report.py --benchmark $BENCH --run-id <id>
```

| Experiment | `BENCH` | `LIST` |
|---|---|---|
| Verified smoke test | `verified` | `instance_lists/verified_smoke_5.txt` |
| Pro smoke test | `pro` | `instance_lists/pro_smoke_10.txt` |
| Pro main run | `pro` | `instance_lists/pro_main_200.txt` (`g_baseline_100` + `g_baseline2_100`) |

`run_batch.py` resumes: rerunning a run id skips completed instances and resumes
the others from v1.5's per-agent caches (`--retry-failed` reruns failures). Use
`--remove-images` when disk is short; evaluation then pulls the images again.

Outputs go to `work/` (gitignored): `work/<benchmark>/retrieval/` (Advisor results
and the history file), `work/<benchmark>/knowledge/` (v1.5 knowledge caches,
shared across runs), `work/<benchmark>/runs/<id>/` (`run.json`,
`predictions.jsonl`, `summary.json`, `cost.json`, `evaluation/`, and per instance
the agents' cached messages, logs and `result.json`).

## What differs from v1.5

| Area | v1.5 | Here | Why |
|---|---|---|---|
| Agent model | `claude-sonnet-4-20250514` | `claude-haiku-4-5-20251001`, same settings (temperature 1, thinking 1024, 3072 output; aggregator 4096 / 8192) | The experiment |
| Text editor tool | `text_editor_20250429` | `text_editor_20250728` (same commands and input schema; `max_characters` not set) | Haiku 4.5 rejects `text_editor_20250429` |
| Knowledge writer | `claude-3-5-sonnet-latest`, temperature 1, 8096 tokens | Haiku 4.5, same settings | Retired; the experiment |
| Historical issues | Offline: an LLM summary of each issue, Qwen3-Embedding-8B top 20, Qwen3-Reranker-4B, top 3 (paper §5.5); the file read is `…-Reranking-Summary-Filtered.jsonl`, filter not described; file not released | Lingxi Advisor 0.8.6 retrieval stage (`scripts/retrieve.py`): GitHub issue search for 10 candidates (Advisor default 5; 20 would double the GitHub GraphQL calls, the binding limit), temporal and leakage checks, Haiku relevance gate; then the top 3 safe candidates by the gate's scores (similarity, then misleading risk, then search order) **without the gate's acceptance threshold**, one issue per fix commit. Instances left with fewer than 3 are searched again with Advisor's "evaluation" strategy (`scripts/advisor_evaluation_search.py`: full closed-issue catalog, every search phase, fixes found through commit messages), which its CLI does not expose. The gate's decision and the strategy are kept in each row | Advisor is the authors' public reimplementation; their data is internal. With Advisor's defaults (5 candidates, threshold) only 1 of 16 candidates was kept on five Pro tasks, so most tasks ran without knowledge; v1.5 always used its reranker's top 3. Still different from v1.5: the pool is GitHub keyword search, not nearest neighbours over the repository's history, and the ranking is Haiku's 0-5 transferability score, not Qwen3-Reranker. The bounded strategy accepts only PR-linked fixes, so NodeBB (fixes closed by commits) got no candidates until the evaluation rerun; some instances still get fewer than 3 when the search itself finds few issues (vuls-0ec945: 4 hits, 2 with a fix) |
| DeepWiki | Self-hosted, answers with `gemini-2.5-flash` (thinking on by default) | DeepWiki-Open `a5f39e3` (2025-07-21), answers with Haiku 4.5 at temperature 0.7, no extended thinking; OpenAI `text-embedding-3-small` embeddings (DeepWiki's default; v1.5's embedder is not recorded) | Claude-only model choice; their server setup was not released |
| Benchmark | SWE-bench Verified only | Also SWE-bench Pro V1 (`ScaleAI/SWE-bench_Pro` config `v1`, 731 tasks); the task text adds Pro's Requirements and Interface sections | The experiment; V1 matches the runner's baselines |
| Verified dataset | `princeton-nlp/SWE-bench_Verified` | `SWE-bench/SWE-bench_Verified@78f471bf` | Identical problem statements, base commits and patches; it has the columns swebench 5 scores with, and it is the runner's |
| Containers | Published image, full network, SWE-ReX over a published port | Prepared image per instance: repository reset to the base commit (Pro: the harness's own reset/checkout, not `before_repo_set_cmd`, which checks out the hidden tests), every commit outside the base commit's ancestry pruned, no network (internal Docker network), SWE-ReX server from a private Python | Pro images carry the fix commit and the upstream is reachable |
| Prediction | `git diff` of tracked files | Also new files; artifacts and v1.5's `reproduction.*` scratch script dropped (runner's filter) | 243 of 731 Pro tasks need a new file |
| File tools | Accept any host path; the agents' container paths (`/testbed/...`, `/app/...`) do not exist on the host, so viewing one raises | Container repository paths are mapped onto the instance's repository copy; any other path gets v1.5's "path does not exist" error as a tool result instead of raising | The host holds datasets with gold patches. Raising there ended the whole instance: Haiku viewed `/` (on the v1.5 host, a listing of the host's root) and `/root` |
| Pager | Not set: in SWE-ReX's terminal, `git log`/`git show` open `less` (or `more`, in 9 of 11 Pro repositories checked) and wait until the shell tool's timeout and retry, about 3 minutes, then return a timeout error | `PAGER=cat GIT_PAGER=cat` in the prepared image | Saves wall time and gives the agents the git output they asked for; it happened twice per instance in the first Verified runs |
| `core.fileMode` | Not set; v1.5's `chmod -R 777` marks every file modified in the agents' `git status` | `false` in the prepared repository | v1.5's patch command already ignored modes |
| DeepWiki repository copy | Taken from the first container started for the instance, which can be a historical checkout | Only from the instance's own checkout; `.git` and `node_modules` (skipped by DeepWiki anyway) not copied | Bug: the agents' wiki could describe old code |
| DeepWiki client | No timeout; `None` on failure | 900 s timeout; an error message on failure; index built before the agents start | A hung server would hang the run |
| Knowledge failures | An error message is cached as knowledge | Retried up to 3 times; then that issue is dropped and its decoder runs without knowledge (v1.5's behaviour for fewer than 3 issues) | Error text is not knowledge |
| Dependencies | No lockfile, SWE-ReX `main` | `uv.lock` resolved as of 2025-07-26, SWE-ReX at that day's `main` | Current LangChain/LangGraph releases break v1.5's imports |

Fixes to the `db17799` commit, which does not run as committed:

* `tool_set/utils.py`: `maybe_truncate` restored. The "update for v1.5" commit
  deleted it while `oheditor.py` still imports it; the restored body is identical
  to the original and to the authors' own restore in `c9e765c`.
* `tool_set/edit_tool.py`: imports of `get_runtime_config` and `logger` added; both
  are used but never defined (every edit or view raised `NameError`).
* `tool_set/dev_knowledge.py`: a history-issue file with no rows (no issue
  selected) reads as having no retrieved issues. v1.5 read one file covering every
  instance; ours is per instance, and an empty one has no `instance_id` column
  (`KeyError`).
* `agent/utils.py` (`compress_agent_thinking_observation_action`): an agent whose
  output is a single message, with no tool call, is summarised as that message's
  text, not its `.content`. For Claude that content can be a list of blocks; the
  solver appends its patch with `+=`, which extended the message's own list with
  one string per character and failed the state reducer (`TypeError`). Haiku's
  solver ended this way on sympy__sympy-20916.
* `tool_set/oheditor.py` (`view` of a directory): the `find` listing is truncated
  with a notice appended to its last line, and every line is then checked with
  `Path.is_dir()`. When that last line exceeds the file-name limit the check
  raises `OSError: File name too long` and ends the instance; the line is now
  treated as a file. Happened on NodeBB (a directory listing of the repository
  root).
* Containers: v1.5 stops an agent's container only when its node succeeds, so a
  node that raises leaves it running; LangGraph's RetryPolicy then starts a new
  one (Haiku's solver once returned an empty message on qutebrowser). When an
  instance ends, `scripts/run_instance.py` now removes every container started
  from that instance's prepared image. The agents' behaviour is unchanged.

## Kept from v1.5 on purpose

* Every agent gets a fresh container; decoders run one after another.
* The solver replays the mapper's file edits but not its shell commands (the
  replay looks for a tool named `run_shell_cmd`; the tool is registered as `bash`).
* Viewing a path inside the repository that does not exist raises, which fails the agent's node; the
  node is retried once (RetryPolicy), then the instance fails.
* The prompts are unchanged, including "I've uploaded the python code repository"
  on Go and JavaScript repositories.
* The shell tool's description still mentions a package mirror; there is none,
  since the containers are offline (the Pro images ship their dependencies).
