#!/usr/bin/env bash
# Third-party components of the SWE-bench reproduction, each pinned and in its own
# environment under external/ (gitignored). Safe to re-run.
#
#   DeepWiki-Open   a5f39e3 (2025-07-21) + patches/deepwiki-open-anthropic.patch
#   Lingxi Advisor  0.8.6 from spine-se-lab/Lingxi-advisor a9d4ded (retrieval only)
#   SWE-bench_Pro-os 66f9276 (v2.0.0; run scripts for all 731 V1 tasks)
#   swebench        5.0.2 (SWE-bench Verified harness)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
EXT="$ROOT/external"
mkdir -p "$EXT"

for tool in uv git docker rg unzip; do
  command -v "$tool" >/dev/null || { echo "missing: $tool (rg is ripgrep, used by v1.5's search tool)"; exit 1; }
done

DEEPWIKI_COMMIT=a5f39e356dbaf38a869af3f4436ba2797123927f
ADVISOR_COMMIT=a9d4ded51fcec6266840ec9a3016c1dd32c60bef
ADVISOR_WHEEL_SHA256=d36fef6a3cc30090509cae7fa1bddf6e4a408f45b0f7f5db24db5ae5f1359682
PRO_OS_COMMIT=66f92766bba642462d4bbe5479e83f91f9211862
SWEBENCH_VERSION=5.0.2

checkout() {  # url dir commit
  [ -d "$2/.git" ] || git clone -q "$1" "$2"
  git -C "$2" fetch -q origin || true
  git -C "$2" reset -q --hard "$3"
  git -C "$2" clean -qfd
}

echo "== DeepWiki-Open"
checkout https://github.com/AsyncFuncAI/deepwiki-open.git "$EXT/deepwiki-open" "$DEEPWIKI_COMMIT"
git -C "$EXT/deepwiki-open" apply "$ROOT/patches/deepwiki-open-anthropic.patch"
[ -x "$EXT/deepwiki-venv/bin/python" ] || uv venv -q --python 3.12 "$EXT/deepwiki-venv"
# Dependencies as of the commit's date.
uv pip install -q --no-config --python "$EXT/deepwiki-venv/bin/python" --exclude-newer 2025-07-22T00:00:00Z \
  -r "$EXT/deepwiki-open/api/requirements.txt"

echo "== Lingxi Advisor"
checkout https://github.com/spine-se-lab/Lingxi-advisor.git "$EXT/Lingxi-advisor" "$ADVISOR_COMMIT"
UPSTREAM="$EXT/Lingxi-advisor/plugins/lingxi-advisor/source/upstream"
rm -rf "$EXT/advisor-extension" && mkdir -p "$EXT/advisor-extension"
unzip -q "$UPSTREAM/lingxi-advisor-extension-0.8.6.zip" -d "$EXT/advisor-extension"
WHEEL="$(ls "$EXT"/advisor-extension/*/runtime/lingxi_advisor-0.8.6-py3-none-any.whl)"
echo "$ADVISOR_WHEEL_SHA256  $WHEEL" | sha256sum -c -
[ -x "$EXT/advisor-venv/bin/python" ] || uv venv -q --python 3.12 "$EXT/advisor-venv"
uv pip install -q --no-config --python "$EXT/advisor-venv/bin/python" -r "$UPSTREAM/requirements.lock"
uv pip install -q --no-config --python "$EXT/advisor-venv/bin/python" --no-deps "$WHEEL"
"$EXT/advisor-venv/bin/lingxi-advisor-candidate-search" --help >/dev/null

echo "== SWE-bench_Pro-os"
checkout https://github.com/scaleapi/SWE-bench_Pro-os.git "$EXT/SWE-bench_Pro-os" "$PRO_OS_COMMIT"
[ -x "$EXT/pro-os-venv/bin/python" ] || uv venv -q --python 3.12 "$EXT/pro-os-venv"
uv pip install -q --no-config --python "$EXT/pro-os-venv/bin/python" -r "$EXT/SWE-bench_Pro-os/requirements.txt"

echo "== swebench $SWEBENCH_VERSION"
[ -x "$EXT/swebench-venv/bin/python" ] || uv venv -q --python 3.12 "$EXT/swebench-venv"
uv pip install -q --no-config --python "$EXT/swebench-venv/bin/python" "swebench==$SWEBENCH_VERSION"

echo "== Lingxi"
uv sync -q
echo "Done. Start DeepWiki with scripts/start_deepwiki.sh, then run scripts/smoke_check.py."
