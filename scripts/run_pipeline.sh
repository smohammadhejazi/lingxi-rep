#!/usr/bin/env bash
# Retrieval, the v1.5 workflow, evaluation and the cost report for one instance
# list, unattended. Every step resumes: a rerun skips what is already done.
#
#   scripts/run_pipeline.sh <benchmark> <instance-list> <run-id> [workers]
#   tmux new -s night -d 'scripts/run_pipeline.sh pro instance_lists/pro_tonight_70.txt pro-main-200 6; exec bash'
#
# Retrieval runs twice (the second pass retries instances whose Advisor run
# failed); instances still without a result run without knowledge. Failed
# instances get one retry before evaluation. The DeepWiki server must be up.

set -u
cd "$(dirname "$0")/.."

BENCH=${1:?benchmark}
LIST=${2:?instance list}
RUN_ID=${3:?run id}
WORKERS=${4:-4}
LOG="work/${BENCH}/pipeline-${RUN_ID}-$(date +%Y%m%dT%H%M%S).log"
mkdir -p "work/${BENCH}"

step() { echo; echo "=== $(date '+%F %T') $*"; }

{
  step "retrieval ($LIST)"
  uv run python scripts/retrieve.py --benchmark "$BENCH" --instances "$LIST" --workers 4
  step "retrieval, second pass"
  uv run python scripts/retrieve.py --benchmark "$BENCH" --instances "$LIST" --workers 4

  step "run_batch ($WORKERS workers, run $RUN_ID)"
  uv run python scripts/run_batch.py --benchmark "$BENCH" --instances "$LIST" --workers "$WORKERS" \
    --run-id "$RUN_ID" --allow-missing-retrieval
  step "run_batch, retry failed instances"
  uv run python scripts/run_batch.py --benchmark "$BENCH" --instances "$LIST" --workers "$WORKERS" \
    --run-id "$RUN_ID" --allow-missing-retrieval --retry-failed

  step "evaluation"
  uv run python scripts/evaluate.py --benchmark "$BENCH" --run-id "$RUN_ID" --workers "$WORKERS"
  step "cost report"
  uv run python scripts/cost_report.py --benchmark "$BENCH" --run-id "$RUN_ID"
  step "done"
} 2>&1 | tee "$LOG"
