"""
Instance containers for the SWE-bench Pro reproduction.

v1.5 started each instance image as published and talked to it over a port
published on the host, with full network access. On SWE-bench Pro that leaks
the answer: the images are full clones that carry the fix commit, and the
upstream repository is one `git fetch` away. This module

* prepares one image per instance (`prepare_image`): the repository is set to the
  base commit exactly as the evaluation harness sets it, every commit that is not
  an ancestor of the base commit is pruned, and the image entrypoint is cleared;
* starts containers with no network (`IsolatedDockerDeployment`): an internal
  Docker network has no route out and publishes no ports, so the SWE-ReX runtime
  is reached on the container's own address;
* runs the SWE-ReX server from a private Python, so it does not depend on the
  image's Python or on installing anything once the container is offline;
* captures the prediction (`capture_patch`) including new files.
"""

import asyncio
import fnmatch
import hashlib
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request

from filelock import FileLock
from swerex.deployment.docker import DockerDeployment
from swerex.runtime.abstract import Command
from swerex.runtime.config import RemoteRuntimeConfig
from swerex.runtime.remote import RemoteRuntime

from src.agent import benchmark
from src.agent.constant import DOCKER_MAP_DIR, REPO_MAP_DIR, RUNTIME_DIR
from src.agent.logging_config import get_logger

logger = get_logger(__name__)

INTERNAL_NETWORK = os.environ.get("LINGXI_DOCKER_NETWORK", "lingxi-isolated")
SERVER_DIR = "/opt/lingxi-python"
SERVER_PYTHON = f"{SERVER_DIR}/bin/python3"
SWEREX_SERVER_VERSION = "1.3.0"
# Dependency cut-off for the SWE-ReX server install, the same date as uv.lock.
SERVER_EXCLUDE_NEWER = "2025-07-26T00:00:00Z"
# python-build-standalone CPython for glibc images, relocatable and built
# against glibc 2.17, so it runs in every glibc-based instance image.
PBS_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/20250723/"
    "cpython-3.11.13%2B20250723-x86_64-unknown-linux-gnu-install_only.tar.gz"
)
PBS_SHA256 = "71c0eb7f5025e2cde7a473b724e2959041fdba243ea58bea71ea2e3455107ea2"
PREP_VERSION = "2"

CACHE_DIR = os.path.join(RUNTIME_DIR, "cache")
HOST_SERVER_DIR = os.path.join(CACHE_DIR, "lingxi-python")
LOCK_DIR = os.path.join(RUNTIME_DIR, "locks")

# Agent-created artifacts and dependency directories that never belong in a
# prediction. Same list as swebench-pro-cli-runner (pipeline/patching.py).
ARTIFACT_PATTERNS = [
    "CLAUDE.md", ".claude", ".claude/*",
    "node_modules", "node_modules/*",
    "__pycache__", "__pycache__/*", "*.pyc", "*.pyo",
    "venv", "venv/*", ".venv", ".venv/*", "*.egg-info", "*.egg-info/*",
    "vendor", "vendor/*",
    ".env",
]
# v1.5's Solution Mapper is told to create `reproduction.py`; on non-Python
# repositories it writes the same script in the repository's language.
REPRODUCTION_SCRIPT = re.compile(r"^reproduction[^/]*$")

PSEUDO_REFS = ("ORIG_HEAD", "FETCH_HEAD", "MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "BISECT_HEAD", "AUTO_MERGE")


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def _lock(name: str) -> FileLock:
    os.makedirs(LOCK_DIR, exist_ok=True)
    return FileLock(os.path.join(LOCK_DIR, f"{name}.lock"))


def image_exists(image: str) -> bool:
    return _run(["docker", "image", "inspect", image]).returncode == 0


def ensure_internal_network() -> None:
    """Create the internal Docker network the agent containers join."""
    if _run(["docker", "network", "inspect", INTERNAL_NETWORK]).returncode == 0:
        return
    with _lock("network"):
        if _run(["docker", "network", "inspect", INTERNAL_NETWORK]).returncode == 0:
            return
        result = _run(["docker", "network", "create", "--internal", INTERNAL_NETWORK])
        if result.returncode != 0:
            raise RuntimeError(f"Could not create network {INTERNAL_NETWORK}: {result.stderr}")


# --- SWE-ReX server --------------------------------------------------------


def ensure_host_server_python() -> str:
    """Standalone CPython with the SWE-ReX server, built once on the host and
    mounted read-only into glibc containers."""
    marker = os.path.join(HOST_SERVER_DIR, ".lingxi-ready")
    if os.path.exists(marker):
        return HOST_SERVER_DIR
    with _lock("server-python"):
        if os.path.exists(marker):
            return HOST_SERVER_DIR
        os.makedirs(CACHE_DIR, exist_ok=True)
        tarball = os.path.join(CACHE_DIR, os.path.basename(PBS_URL).replace("%2B", "+"))
        if not os.path.exists(tarball):
            logger.info(f"Downloading standalone Python: {PBS_URL}")
            urllib.request.urlretrieve(PBS_URL, tarball + ".part")
            os.replace(tarball + ".part", tarball)
        with open(tarball, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        if digest != PBS_SHA256:
            raise RuntimeError(f"Checksum mismatch for {tarball}: {digest}")
        staging = tempfile.mkdtemp(dir=CACHE_DIR)
        with tarfile.open(tarball) as tar:
            tar.extractall(staging)
        shutil.rmtree(HOST_SERVER_DIR, ignore_errors=True)
        os.replace(os.path.join(staging, "python"), HOST_SERVER_DIR)
        shutil.rmtree(staging, ignore_errors=True)
        python = os.path.join(HOST_SERVER_DIR, "bin", "python3")
        uv = shutil.which("uv")
        if uv:
            install = [uv, "pip", "install", "--python", python, "--exclude-newer", SERVER_EXCLUDE_NEWER]
        else:
            install = [python, "-m", "pip", "install", "--no-cache-dir"]
        result = _run([*install, f"swe-rex=={SWEREX_SERVER_VERSION}"])
        if result.returncode != 0:
            raise RuntimeError(f"Could not install the SWE-ReX server: {result.stderr}")
        _run([python, "-m", "compileall", "-q", os.path.join(HOST_SERVER_DIR, "lib")])
        with open(marker, "w") as f:
            f.write(SWEREX_SERVER_VERSION + "\n")
    return HOST_SERVER_DIR


def server_constraints() -> str:
    """The host server's package versions, so a musl image gets the same ones."""
    python = os.path.join(ensure_host_server_python(), "bin", "python3")
    return _run([python, "-m", "pip", "freeze", "--disable-pip-version-check"]).stdout


# --- prepared images -------------------------------------------------------


def _image_libc(image: str) -> str:
    result = _run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "/bin/sh", image, "-c",
                   "ldd --version 2>&1 | head -1"])
    return "musl" if "musl" in (result.stdout + result.stderr).lower() else "glibc"


def _git_prep_script(instance: dict) -> str:
    """Repository at the base commit, the image's history pruned to the base
    commit's ancestry. Pro: `git reset --hard` + `git checkout` of the base
    commit, the two steps the SWE-bench Pro harness runs before applying a patch
    (not `before_repo_set_cmd`, whose last line checks the hidden tests out of
    the fix commit). Verified: the repository is left as v1.5 found it."""
    lines = [
        "set -e",
        f"cd {benchmark.REPO_DIR}",
        "git config --global --add safe.directory '*'",
        "git config gc.auto 0",
        # v1.5 runs `chmod -R 777` on the repository when it loads an instance;
        # without this every file shows as modified in the agents' git status.
        # v1.5's own patch command already ignored file modes.
        "git config core.fileMode false",
    ]
    if benchmark.BENCHMARK == benchmark.PRO:
        base = instance["base_commit"]
        lines += [f"git reset -q --hard {base}", f"git checkout -q {base}"]
    lines += [
        # Untracked files the image already has are excluded, so capturing new
        # files cannot put build products into a prediction.
        "mkdir -p .git/info",
        "git ls-files -z --others --exclude-standard | tr '\\0' '\\n' | sed 's|^|/|' >> .git/info/exclude",
        # Every ref but the checked-out branch goes, then whatever only they
        # reached. Same steps as swebench-pro-cli-runner (git_prune_script).
        'gitdir="$(git rev-parse --git-dir)"',
        'branch="$(git symbolic-ref -q HEAD || true)"',
        'if [ -z "$branch" ]; then',
        "  branch=refs/heads/base",
        '  git update-ref "$branch" HEAD',
        '  git symbolic-ref HEAD "$branch"',
        "fi",
        'for remote in $(git remote); do git remote remove "$remote"; done',
        "git for-each-ref --format='delete %(refname)' | grep -vxF \"delete $branch\" | git update-ref --no-deref --stdin",
        'find "$gitdir/refs" "$gitdir/logs/refs" -type f ! -path "$gitdir/$branch" '
        '! -path "$gitdir/logs/$branch" -delete 2>/dev/null || true',
        "for ref in " + " ".join(PSEUDO_REFS) + '; do rm -f "$gitdir/$ref"; done',
        "git reflog expire --expire=now --all",
        "git gc -q --prune=now",
    ]
    return "\n".join(lines) + "\n"


def _musl_server_script() -> str:
    return "\n".join([
        "set -e",
        'PY=""',
        "for c in python3.13 python3.12 python3.11 python3.10 python3; do",
        '  if command -v "$c" >/dev/null 2>&1 && "$c" -c '
        "'import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then",
        '    PY="$(command -v "$c")"; break',
        "  fi",
        "done",
        '[ -n "$PY" ] || { echo "no Python >= 3.10 with venv in this image" >&2; exit 42; }',
        f'"$PY" -m venv {SERVER_DIR}',
        f"{SERVER_DIR}/bin/pip install --no-cache-dir --disable-pip-version-check "
        f"-c /tmp/lingxi/constraints.txt swe-rex=={SWEREX_SERVER_VERSION}",
    ]) + "\n"


def _verify_script(instance: dict) -> str:
    fix = benchmark.fix_commit(instance) or ""
    return "\n".join([
        f"cd {benchmark.REPO_DIR}",
        'echo "head=$(git rev-parse HEAD)"',
        'echo "remotes=$(git remote | wc -l)"',
        'echo "outside=$(git rev-list --all --not HEAD | wc -l)"',
        f'echo "base_ancestor=$(git merge-base --is-ancestor {instance["base_commit"]} HEAD && echo yes || echo no)"',
        f'if [ -n "{fix}" ]; then git cat-file -e "{fix}^{{commit}}" 2>/dev/null && echo fix=present || echo fix=absent; fi',
        'echo "dirty=$(git status --porcelain | wc -l)"',
    ]) + "\n"


def _verify(image: str, instance: dict) -> dict:
    result = _run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "/bin/sh",
                   *server_mount_args(image), image, "-c",
                   _verify_script(instance) + f'{SERVER_PYTHON} -c "import swerex.server" && echo server=ok'])
    facts = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    problems = []
    if facts.get("remotes") != "0":
        problems.append("remotes left")
    if facts.get("outside") != "0":
        problems.append("commits outside the base ancestry")
    if facts.get("base_ancestor") != "yes":
        problems.append("base commit not in history")
    if facts.get("fix") == "present":
        problems.append("fix commit still present")
    if benchmark.BENCHMARK == benchmark.PRO:
        if facts.get("head") != instance["base_commit"]:
            problems.append(f"HEAD is {facts.get('head')}, not the base commit")
        if facts.get("dirty") != "0":
            problems.append("working tree not clean")
    if facts.get("server") != "ok":
        problems.append("SWE-ReX server not runnable")
    facts["problems"] = problems
    if problems:
        logger.error(f"Prepared image check failed for {image}: {problems}\n{result.stderr}")
    return facts


def server_mount_args(image: str) -> list[str]:
    """`docker run` arguments that give a container its SWE-ReX server."""
    label = _run(["docker", "image", "inspect", "-f", '{{index .Config.Labels "lingxi.server"}}', image]).stdout.strip()
    if label == "mount":
        return ["-v", f"{ensure_host_server_python()}:{SERVER_DIR}:ro"]
    return []


def prepared_image(instance: dict) -> str:
    """Local tag of the instance's prepared image; changes with PREP_VERSION."""
    instance_id = instance["instance_id"]
    readable = re.sub(r"[^a-z0-9_.-]", "-", instance_id.lower())[:80]
    digest = hashlib.sha1(f"{benchmark.BENCHMARK}|{instance_id}|{PREP_VERSION}".encode()).hexdigest()[:10]
    return f"lingxi-prepared:{readable}-p{PREP_VERSION}-{digest}"


def prepare_image(instance: dict, force: bool = False) -> str:
    """Build the instance's prepared image once and return its tag."""
    image = prepared_image(instance)
    if image_exists(image) and not force:
        return image
    with _lock(hashlib.sha1(image.encode()).hexdigest()[:16]):
        if image_exists(image) and not force:
            return image
        source = benchmark.source_image(instance)
        if not image_exists(source):
            logger.info(f"Pulling {source}")
            pulled = _run(["docker", "pull", source])
            if pulled.returncode != 0:
                raise RuntimeError(f"Could not pull {source}: {pulled.stderr}")
        libc = _image_libc(source)
        os.makedirs(CACHE_DIR, exist_ok=True)
        context = tempfile.mkdtemp(dir=CACHE_DIR)
        try:
            with open(os.path.join(context, "prep.sh"), "w") as f:
                f.write(_git_prep_script(instance))
            dockerfile = [f"FROM {source}", "USER root", "COPY prep.sh /tmp/lingxi/prep.sh",
                          "RUN /bin/sh /tmp/lingxi/prep.sh && rm -rf /tmp/lingxi"]
            if libc == "musl":
                with open(os.path.join(context, "server.sh"), "w") as f:
                    f.write(_musl_server_script())
                with open(os.path.join(context, "constraints.txt"), "w") as f:
                    f.write(server_constraints())
                dockerfile += ["COPY server.sh constraints.txt /tmp/lingxi/",
                               "RUN /bin/sh /tmp/lingxi/server.sh && rm -rf /tmp/lingxi"]
                server = "image"
            else:
                ensure_host_server_python()
                server = "mount"
            dockerfile += [
                "ENTRYPOINT []",
                "CMD []",
                f'LABEL lingxi.server="{server}" lingxi.prep="{PREP_VERSION}" '
                f'lingxi.instance_id="{instance["instance_id"]}" lingxi.source_image="{source}"',
            ]
            with open(os.path.join(context, "Dockerfile"), "w") as f:
                f.write("\n".join(dockerfile) + "\n")
            logger.info(f"Building {image} from {source} ({libc})")
            built = _run(["docker", "build", "-q", "-t", image, context])
            if built.returncode != 0:
                raise RuntimeError(f"Could not build {image}: {built.stderr[-4000:]}")
        finally:
            shutil.rmtree(context, ignore_errors=True)
        facts = _verify(image, instance)
        if facts["problems"]:
            _run(["docker", "rmi", "-f", image])
            raise RuntimeError(f"Prepared image {image} failed its checks: {facts['problems']}")
        return image


def remove_images(instance: dict, source_too: bool = False) -> None:
    images = [prepared_image(instance)]
    if source_too:
        images.append(benchmark.source_image(instance))
    _run(["docker", "rmi", "-f", *images])


# --- no-network deployment --------------------------------------------------


class IsolatedDockerDeployment(DockerDeployment):
    """DockerDeployment whose container has no network access.

    The container joins an internal Docker network and the runtime is reached on
    the container's address, since internal networks do not publish ports. The
    SWE-ReX server runs from the private Python under `SERVER_DIR`.
    """

    def _get_swerex_start_cmd(self, token: str) -> list[str]:
        return ["/bin/sh", "-c", f"{SERVER_PYTHON} -m swerex.server --auth-token {token}"]

    async def _container_address(self, timeout: float) -> str:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._container_process.poll() is not None:
                raise RuntimeError(
                    f"Container {self._container_name} exited: {self._container_process.stderr.read().decode()}"
                )
            inspected = _run(["docker", "inspect", "-f",
                              "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", self._container_name])
            address = inspected.stdout.strip()
            if inspected.returncode == 0 and address:
                return address
            await asyncio.sleep(0.5)
        raise TimeoutError(f"Container {self._container_name} got no address within {timeout}s")

    async def start(self):
        """Starts the runtime."""
        ensure_internal_network()
        self._pull_image()
        assert self._container_name is None
        self._container_name = self._get_container_name()
        token = self._get_token()
        platform_arg = []
        if self._config.platform is not None:
            platform_arg = ["--platform", self._config.platform]
        rm_arg = ["--rm"] if self._config.remove_container else []
        cmds = [
            "docker",
            "run",
            *rm_arg,
            "--network",
            INTERNAL_NETWORK,
            *platform_arg,
            *self._config.docker_args,
            "--name",
            self._container_name,
            self._config.image,
            *self._get_swerex_start_cmd(token),
        ]
        self.logger.info(f"Starting container {self._container_name} with image {self._config.image} (no network)")
        self._container_process = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._hooks.on_custom_step("Starting runtime")
        address = await self._container_address(timeout=self._config.startup_timeout)
        self._runtime = RemoteRuntime.from_config(
            RemoteRuntimeConfig(host=f"http://{address}", port=8000, timeout=self._runtime_timeout, auth_token=token)
        )
        t0 = time.time()
        await self._wait_until_alive(timeout=self._config.startup_timeout)
        self.logger.info(f"Runtime started in {time.time() - t0:.2f}s")


# --- predictions -------------------------------------------------------------


def _patch_path(header: str) -> str:
    match = re.match(r'diff --git (?:"?a/)(.+?)"? (?:"?b/)(.+?)"?$', header)
    return match.group(2) if match else header


def filter_patch(patch: str) -> str:
    """Drop agent artifacts, symlinks and v1.5's reproduction scripts from a diff."""
    if not patch or not patch.strip():
        return ""
    chunks = re.split(r"(?m)^(?=diff --git )", patch)
    kept = []
    for chunk in chunks:
        if not chunk.startswith("diff --git "):
            continue
        header, _, body = chunk.partition("\n")
        path = _patch_path(header)
        if re.search(r"(?m)^(new|deleted) file mode 120000$", body):
            continue
        if any(fnmatch.fnmatch(path, pattern) for pattern in ARTIFACT_PATTERNS):
            continue
        if re.search(r"(?m)^new file mode ", body) and REPRODUCTION_SCRIPT.match(os.path.basename(path)):
            continue
        kept.append(chunk if chunk.endswith("\n") else chunk + "\n")
    return "".join(kept)


def capture_patch(rc) -> str:
    """The prediction: every change against the commit the instance was loaded
    at, new files included. Runs as its own command, not in the agent's shell
    session, so the session's state cannot affect it."""
    command = (
        f"git add -N . && git -c core.fileMode=false diff {rc.diff_base or 'HEAD'} --no-color --no-ext-diff"
    )
    response = asyncio.run(
        rc.swe_rex_deployment.runtime.execute(
            Command(command=command, shell=True, cwd=benchmark.REPO_DIR, timeout=600)
        )
    )
    if response.exit_code not in (0, None):
        logger.error(f"Patch capture failed ({response.exit_code}): {response.stderr}")
    return filter_patch(response.stdout)


# --- workspace cleanup -------------------------------------------------------


def instance_workspace_dir(instance_id: str) -> str:
    return os.path.join(DOCKER_MAP_DIR, instance_id)


def cleanup_workspaces(instance: dict) -> None:
    """Remove the repository copies the instance's containers left on the host.
    They hold files root created inside the container, so the removal runs in a
    container too."""
    workspace = instance_workspace_dir(instance["instance_id"])
    if os.path.isdir(workspace):
        image = prepared_image(instance)
        if image_exists(image):
            _run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "/bin/sh",
                  "-v", f"{workspace}:/workspace", image, "-c", "rm -rf /workspace/* /workspace/.[!.]* 2>/dev/null; true"])
        shutil.rmtree(workspace, ignore_errors=True)
    shutil.rmtree(os.path.join(REPO_MAP_DIR, instance["instance_id"]), ignore_errors=True)


def describe() -> dict:
    """Sandbox settings, for run metadata."""
    return {
        "network": f"docker internal network '{INTERNAL_NETWORK}' (no external access)",
        "swerex_server": SWEREX_SERVER_VERSION,
        "server_python": PBS_URL,
        "prep_version": PREP_VERSION,
        "artifact_patterns": ARTIFACT_PATTERNS,
        "reproduction_script_pattern": REPRODUCTION_SCRIPT.pattern,
    }
