import asyncio
from typing import Tuple, Any
from datasets.arrow_dataset import shutil
from swerex.runtime.abstract import CreateBashSessionRequest, BashAction, Command, WriteFileRequest
from swerex.deployment.docker import DockerDeployment
import os
import uuid
from swerex.deployment.config import DockerDeploymentConfig
from src.agent import benchmark, sandbox
from src.agent.constant import DOCKER_MAP_DIR, REPO_MAP_DIR


def extract_git_diff_swerex_container(runtime_config_obj=None):
    # Import locally to avoid circular import
    from src.agent.runtime_config import RuntimeConfig, RuntimeType
    
    # Use provided runtime_config if available, otherwise create a new one
    rc = runtime_config_obj if runtime_config_obj is not None else RuntimeConfig()
    
    print("Extracting git diff from SWEREX container")
    
    if not rc.initialized:
        print("ERROR: RuntimeConfig is not initialized")
        return ""
        
    # Compare by int value instead of direct enum comparison for stability
    if int(rc.runtime_type) != int(RuntimeType.SWEREX):
        print(f"ERROR: Expected RuntimeType.SWEREX (value {int(RuntimeType.SWEREX)}), got {rc.runtime_type} (value {int(rc.runtime_type)})")
        return ""
        
    if not rc.swe_rex_deployment:
        print("ERROR: No SWE-REX deployment available")
        return ""
    
    try:
        swe_rex_runtime = rc.swe_rex_deployment.runtime
        
        # First ensure we're in the right directory
        print(f"Running 'cd {benchmark.REPO_DIR}'")
        cd_result = asyncio.run(swe_rex_runtime.run_in_session(
            BashAction(command=f"cd {benchmark.REPO_DIR}", check="ignore")
        ))
        print(f"cd result: {cd_result.exit_code}")
        
        # Make sure all files are added to git tracking
        print("Running 'git add -A'")
        add_result = asyncio.run(swe_rex_runtime.run_in_session(
            BashAction(command="git add -A", check="ignore")
        ))
        print(f"git add result: {add_result.exit_code}")
        
        # Get the diff
        print("Running git diff")
        git_diff_result = asyncio.run(swe_rex_runtime.run_in_session(
            BashAction(command="git -c core.fileMode=false diff --exit-code --cached --no-color", check="ignore")
        ))
        print(f"Git diff result: '{git_diff_result.output}'")

        patch = git_diff_result.output
        normalized_patch = "\n".join(patch.splitlines())
        # add a new line to the patch
        normalized_patch = normalized_patch + "\n"
        return normalized_patch
        
    except Exception as e:
        print(f"ERROR in extract_git_diff_swerex_container: {e}")
        return ""

async def load_swe_instance_for_swerex(instance_id: str,checkout_commit: str | None = None) -> Tuple[DockerDeployment, str, str]:
    instance = benchmark.get_instance(instance_id)
    # The instance image with its history pruned to the base commit, started
    # with no network access (see src/agent/sandbox.py).
    docker_image_name = sandbox.prepare_image(instance)
    repo_dir = benchmark.REPO_DIR
    repo_name = os.path.basename(repo_dir)

    # Create a unique local directory to map into the container
    tmp_folder_name = str(uuid.uuid4())[:8]
    docker_map_path = os.path.join(sandbox.instance_workspace_dir(instance_id), tmp_folder_name)
    os.makedirs(docker_map_path, exist_ok=True)
    print(f"docker_map_path: {docker_map_path}")
    # Prepare docker_args for volume mapping
    docker_args = [
        "-v", f"{docker_map_path}:/docker_map",
        *sandbox.server_mount_args(docker_image_name),
    ]
    # You can add more docker_args as needed, e.g. user, etc.

    deployment = sandbox.IsolatedDockerDeployment(
        image=docker_image_name,
        docker_args=docker_args,
        pull="never",
    )
    await deployment.start()

    swe_rex_runtime = deployment.runtime

    await swe_rex_runtime.create_session(CreateBashSessionRequest())
    print(await swe_rex_runtime.run_in_session(BashAction(command=f"cd {repo_dir}")))

    if checkout_commit:
        print(await swe_rex_runtime.run_in_session(BashAction(command=f"git checkout {checkout_commit}", check="ignore")))

    print(await swe_rex_runtime.run_in_session(BashAction(command="git config user.name 'Temp User' && git config user.email 'temp@example.com' && git commit -am 'swe-bench-extra'", check="ignore")))
    # The commit the prediction is diffed against.
    diff_base = (await swe_rex_runtime.run_in_session(BashAction(command="git rev-parse HEAD"))).output.strip()
    print(await swe_rex_runtime.run_in_session(BashAction(command="cd /")))
    print(await swe_rex_runtime.run_in_session(BashAction(command=f"mv {repo_dir}/ /docker_map/")))
    print(await swe_rex_runtime.run_in_session(BashAction(command=f"chmod -R 777 /docker_map/{repo_name}")))
    print(await swe_rex_runtime.run_in_session(BashAction(command=f"ln -s /docker_map/{repo_name} {repo_dir}")))
    print(await swe_rex_runtime.run_in_session(BashAction(command=f"cd {repo_dir}")))

    project_path = os.path.join(docker_map_path, repo_name)

    # Ensure a pristine copy in repo_map (for wiki). Only from the instance's own
    # checkout: a historical checkout (checkout_commit) must not become the code
    # DeepWiki answers questions about. Directories DeepWiki skips anyway are not
    # copied, and symlinks are copied as links.
    repo_map_path = os.path.join(REPO_MAP_DIR, f"{instance_id}")
    docker_map_repo_path = os.path.join(docker_map_path, repo_name)
    if checkout_commit is None and not os.path.exists(repo_map_path):
        print(f"Copying pristine copy of {instance_id} to {repo_map_path} for wiki")
        shutil.copytree(
            docker_map_repo_path,
            repo_map_path,
            symlinks=True,
            ignore=shutil.ignore_patterns(".git", "node_modules"),
        )

    print(f"Project path: {project_path}")
    return deployment, project_path, diff_base


    