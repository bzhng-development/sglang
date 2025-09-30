import json
import modal
import shutil
import subprocess
import time
import sys
from pathlib import Path
from typing import Iterable, Optional


MODEL_REGISTRY = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
]


def _print_model_registry():
    print("Available Hugging Face models configured for the shared Modal Volume:")
    for model in MODEL_REGISTRY:
        print(f"- {model}")


MODEL_STORE_VOLUME = modal.Volume.from_name(
    "sglang-model-store",
    create_if_missing=True,
    version=2,
)

MODEL_DIR = Path("/models").as_posix()

MODEL_DIR_PATH = Path(MODEL_DIR)

SHARED_SECRETS = [
    modal.Secret.from_name("sglang-gh"),
    modal.Secret.from_name("sglang-hf"),
]

# GPU pricing information (per GPU per hour)
GPU_PRICING = {
    "T4": 0.59,
    "L4": 0.80,
    "A10": 1.10,
    "A100-40GB": 2.10,
    "A100-80GB": 2.50,
    "L40S": 1.95,
    "H100": 3.95,
    "H200": 4.54,
    "B200": 6.25,
}

# Available GPU configurations
GPU_CONFIGS = {
    "T4": [1, 2, 4],
    "L4": [1, 2, 4, 8],
    "A10": [1, 2, 4],
    "A100": [1, 2, 4, 8],  # Includes both 40GB and 80GB variants
    "L40S": [1, 2, 4],
    "H100": [1, 2, 4, 8],
    "H200": [1, 2, 4, 8],
    "B200": [1, 2, 4, 8],
}

def _install_lazygit():
    """Download and install the latest lazygit release."""
    import json
    import urllib.request
    import subprocess
    import shutil
    from pathlib import Path
    
    # Fetch the latest release info from GitHub
    url = "https://api.github.com/repos/jesseduffield/lazygit/releases/latest"
    with urllib.request.urlopen(url) as response:
        data = json.load(response)
        version = data["tag_name"].lstrip("v")
    
    print(f"Installing lazygit version {version}")
    
    # Download the tarball
    tarball_url = f"https://github.com/jesseduffield/lazygit/releases/download/v{version}/lazygit_{version}_Linux_x86_64.tar.gz"
    tarball_path = "/tmp/lazygit.tar.gz"
    
    with urllib.request.urlopen(tarball_url) as response:
        with open(tarball_path, 'wb') as f:
            f.write(response.read())
    
    # Extract and install
    subprocess.run(["tar", "-C", "/tmp", "-xf", tarball_path, "lazygit"], check=True)
    
    # Move to /usr/local/bin
    shutil.move("/tmp/lazygit", "/usr/local/bin/lazygit")
    subprocess.run(["chmod", "+x", "/usr/local/bin/lazygit"], check=True)
    
    print(f"✓ lazygit {version} installed successfully")

def show_help():
    """Display help information with GPU pricing and configurations."""
    print("SGLang SSH Modal Functions - GPU Pricing Information")
    print("=" * 55)
    print()
    print("Available GPU types and pricing (per GPU per hour):")
    print("-" * 50)
    for gpu_type, price in GPU_PRICING.items():
        print(f"Nvidia {gpu_type:<17} ${price:.2f} / h")
    
    print()
    print("Available configurations:")
    for gpu_type, counts in GPU_CONFIGS.items():
        count_str = ", ".join(f"{c}x" for c in counts)
        if gpu_type == "A100":
            print(f"- {gpu_type} (all variants): {count_str}")
        else:
            print(f"- {gpu_type}: {count_str}")
    
    print("- Total cost = (price per GPU) × (number of GPUs) × (hours used)")
    
    print()
    print("Example usage:")
    print("  modal run ssh.py::sglang_h100_4  # 4x H100 GPUs")
    print("  modal run ssh.py::sglang_a10_1   # 1x A10 GPU")

    print()
    _print_model_registry()
    print()
    print("Utilities baked into container:")
    print("- GitHub CLI (gh)")
    print("- uv Python manager")
    print("- LazyGit")
    print("(Use Modal secrets for GH_TOKEN; not baked into image.)")
    print()
    print("Note: H100! specifically requests H100 to avoid auto-upgrade to H200")
    print()
    print("A100 GPUs")
    print()
    print("A100s are based on NVIDIA's Ampere architecture. Modal offers two versions")
    print("of the A100: one with 40 GB of RAM and another with 80 GB of RAM.")
    print()
    print("To request an A100 with 40 GB of GPU memory, use gpu=\"A100\":")
    print("Modal may automatically upgrade a gpu=\"A100\" request to run on an 80 GB A100.")
    print("This automatic upgrade does not change the cost of the GPU.")
    print()
    print("You can specifically request a 40GB A100 with the string A100-40GB.")
    print("To specifically request an 80 GB A100, use the string A100-80GB.")

# Modal setup
sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:dev", add_python="3.12")
    .run_commands(
        "apt-get update && apt-get install -y git clang openssh-server sudo wget curl tar",
    )
    .run_commands(
        "mkdir -p -m 755 /etc/apt/keyrings",
        "mkdir -p -m 755 /etc/apt/sources.list.d",
    )
    .run_commands(
        "wget -nv -O /tmp/githubcli-archive-keyring.gpg https://cli.github.com/packages/githubcli-archive-keyring.gpg",
    )
    .run_commands(
        "cp /tmp/githubcli-archive-keyring.gpg /etc/apt/keyrings/githubcli-archive-keyring.gpg",
        "chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg",
    )
    .run_commands(
        'bash -c "echo \'deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main\' > /etc/apt/sources.list.d/github-cli.list"',
    )
    .run_commands("apt-get update")
    .run_commands("apt-get install -y gh")
    .run_commands(
        "mkdir -p /run/sshd /root/.ssh && chmod 700 /root/.ssh",
    )
    # Replace with your actual public key path if different:
    .add_local_file("/Users/vincentzed/.ssh/id_rsa.pub", "/root/.ssh/id_rsa.pub", copy=True)
    .run_commands(
        "cat /root/.ssh/*.pub > /root/.ssh/authorized_keys",
        "chmod 600 /root/.ssh/authorized_keys"
    )
    # Harden sshd: key-only auth, no passwords
    .run_commands(
        "echo 'PasswordAuthentication no' >> /etc/ssh/sshd_config",
        "echo 'PermitRootLogin prohibit-password' >> /etc/ssh/sshd_config",
        "echo 'PubkeyAuthentication yes' >> /etc/ssh/sshd_config",
    )
    # PAM tweak often recommended in containers to avoid noisy logs
    .run_commands("sed -i 's@session\\s*required\\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd")
    .run_function(_install_lazygit)
    .run_commands("git clone https://github.com/bzhng-development/sglang.git", force_build=False)
    .pip_install("uv")
    .run_commands("cd sglang && uv pip install -e 'python[all]' --system", force_build=False)
)

download_image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

llama4_maverick_fp8 = modal.Volume.from_name("llama-4-maverick-17b-fp8")

app = modal.App("sglang-ssh", image=sglang_image)


def _ensure_model_dir():
    MODEL_DIR_PATH.mkdir(parents=True, exist_ok=True)


def _resolve_model_path(model_id: str) -> Path:
    return MODEL_DIR_PATH / model_id


@app.function(image=download_image, volumes={MODEL_DIR: MODEL_STORE_VOLUME}, timeout=86400, secrets=SHARED_SECRETS)
def list_models() -> list[str]:
    _ensure_model_dir()
    MODEL_STORE_VOLUME.reload()
    models = sorted(
        str(path.relative_to(MODEL_DIR_PATH))
        for path in MODEL_DIR_PATH.glob("*")
        if path.is_dir()
    )

    if models:
        print("Models available in shared volume:")
        for name in models:
            print(f"- {name}")
    else:
        print("No models found in shared volume. Use download_model or seed_qwen_models to populate it.")

    return models


@app.function(image=download_image, volumes={MODEL_DIR: MODEL_STORE_VOLUME}, timeout=86400, secrets=SHARED_SECRETS)
def download_model(repo_id: str, revision: Optional[str] = None, force: bool = False) -> str:
    from huggingface_hub import snapshot_download

    _ensure_model_dir()
    MODEL_STORE_VOLUME.reload()
    target_dir = _resolve_model_path(repo_id)

    if target_dir.exists() and not force:
        message = f"Model already present at {target_dir}"
        print(message)
        return message

    print(f"Downloading {repo_id} into {target_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        revision=revision,
    )
    MODEL_STORE_VOLUME.commit()
    message = f"Downloaded {repo_id} to {target_dir}"
    print(message)
    return message


@app.function(image=download_image, volumes={MODEL_DIR: MODEL_STORE_VOLUME}, timeout=86400, secrets=SHARED_SECRETS)
def seed_qwen_models(force: bool = False) -> list[str]:
    """Download all Qwen models in MODEL_REGISTRY into the shared volume."""
    results = []
    for repo_id in MODEL_REGISTRY:
        result = download_model.remote(repo_id=repo_id, force=force)
        results.append(result)
    return results


@app.function(image=download_image, volumes={MODEL_DIR: MODEL_STORE_VOLUME}, secrets=SHARED_SECRETS)
def delete_model(repo_ids: Optional[str] = None, repo_id: Optional[str] = None) -> list[str]:
    """Remove one or more model directories from the shared volume."""
    _ensure_model_dir()
    MODEL_STORE_VOLUME.reload()

    targets: list[str] = []
    if repo_ids:
        if isinstance(repo_ids, str):
            try:
                parsed = json.loads(repo_ids)
                if isinstance(parsed, str):
                    targets.append(parsed)
                elif isinstance(parsed, Iterable):
                    for item in parsed:
                        if isinstance(item, str):
                            targets.append(item)
            except json.JSONDecodeError:
                for token in repo_ids.split(","):
                    token = token.strip()
                    if token:
                        targets.append(token)
    if repo_id:
        targets.append(repo_id)

    if not targets:
        message = "No repo ids provided"
        print(message)
        return [message]

    results = []
    for repo in targets:
        path = _resolve_model_path(repo)
        if not path.exists():
            msg = f"No model found at {path}"
            print(msg)
            results.append(msg)
            continue

        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

        msg = f"Removed {repo} from shared volume"
        print(msg)
        results.append(msg)

    MODEL_STORE_VOLUME.commit()
    return results

def ssh_setup():
    """Helper function to handle SSH setup and keep container alive."""
    _ensure_model_dir()
    # Start sshd in the background
    subprocess.Popen(["/usr/sbin/sshd", "-D", "-e"])
    # Expose port 22 to the public Internet as a raw TCP socket; SSH provides encryption.
    with modal.forward(port=22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print(f"ssh -o StrictHostKeyChecking=no -p {port} root@{host}")
        # Keep the container alive for 24 hours
        time.sleep(86400)

# T4 GPU configurations
@app.function(
    gpu="t4:1",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    secrets=SHARED_SECRETS,
    timeout=86400,
)
def sglang_t4_1():
    ssh_setup()

@app.function(
    gpu="t4:2",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    secrets=SHARED_SECRETS,
    timeout=86400,
)
def sglang_t4_2():
    ssh_setup()

@app.function(
    gpu="t4:4",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    secrets=SHARED_SECRETS,
    timeout=86400,
)
def sglang_t4_4():
    ssh_setup()

# L4 GPU configurations
@app.function(
    gpu="l4:1",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    secrets=SHARED_SECRETS,
    timeout=86400,
)
def sglang_l4_1():
    ssh_setup()

@app.function(
    gpu="l4:2",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    secrets=SHARED_SECRETS,
    timeout=86400,
)
def sglang_l4_2():
    ssh_setup()

@app.function(
    gpu="l4:4",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    secrets=SHARED_SECRETS,
    timeout=86400,
)
def sglang_l4_4():
    ssh_setup()

@app.function(
    gpu="l4:8",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    secrets=SHARED_SECRETS,
    timeout=86400,
)
def sglang_l4_8():
    ssh_setup()

# A10 GPU configurations
@app.function(
    gpu="a10:1",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    secrets=SHARED_SECRETS,
    timeout=86400,
)
def sglang_a10_1():
    ssh_setup()

@app.function(
    gpu="a10:2",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    secrets=SHARED_SECRETS,
    timeout=86400,
)
def sglang_a10_2():
    ssh_setup()

@app.function(
    gpu="a10:4",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    secrets=SHARED_SECRETS,
    timeout=86400,
)
def sglang_a10_4():
    ssh_setup()

# A100 GPU configurations (standard - may auto-upgrade to 80GB)
@app.function(
    gpu="a100:1",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    secrets=SHARED_SECRETS,
    timeout=86400,
)
def sglang_a100_1():
    ssh_setup()

@app.function(
    gpu="a100:2",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    secrets=SHARED_SECRETS,
    timeout=86400,
)
def sglang_a100_2():
    ssh_setup()

@app.function(
    gpu="a100:4",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_a100_4():
    ssh_setup()

@app.function(
    gpu="a100:8",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_a100_8():
    ssh_setup()

# A100-40GB GPU configurations (specifically 40GB)
@app.function(
    gpu="A100-40GB:1",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_a100_40gb_1():
    ssh_setup()

@app.function(
    gpu="A100-40GB:2",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_a100_40gb_2():
    ssh_setup()

@app.function(
    gpu="A100-40GB:4",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_a100_40gb_4():
    ssh_setup()

@app.function(
    gpu="A100-40GB:8",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_a100_40gb_8():
    ssh_setup()

# A100-80GB GPU configurations (specifically 80GB)
@app.function(
    gpu="A100-80GB:1",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_a100_80gb_1():
    ssh_setup()

@app.function(
    gpu="A100-80GB:2",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_a100_80gb_2():
    ssh_setup()

@app.function(
    gpu="A100-80GB:4",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_a100_80gb_4():
    ssh_setup()

@app.function(
    gpu="A100-80GB:8",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_a100_80gb_8():
    ssh_setup()

# L40S GPU configurations
@app.function(
    gpu="l40s:1",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_l40s_1():
    ssh_setup()

@app.function(
    gpu="l40s:2",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_l40s_2():
    ssh_setup()

@app.function(
    gpu="l40s:4",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_l40s_4():
    ssh_setup()

# H100 GPU configurations
@app.function(
    gpu="h100:1",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_h100_1():
    ssh_setup()

@app.function(
    gpu="h100:2",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_h100_2():
    ssh_setup()

@app.function(
    gpu="h100:4",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_h100_4():
    ssh_setup()

@app.function(
    gpu="h100:8",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_h100_8():
    ssh_setup()

# H100! GPU configurations (specifically H100, no auto-upgrade)
@app.function(
    gpu="H100!:1",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_h100_pin_1():
    ssh_setup()

@app.function(
    gpu="H100!:2",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_h100_pin_2():
    ssh_setup()

@app.function(
    gpu="H100!:4",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_h100_pin_4():
    ssh_setup()

@app.function(
    gpu="H100!:8",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_h100_pin_8():
    ssh_setup()

# H200 GPU configurations
@app.function(
    gpu="h200:1",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_h200_1():
    ssh_setup()

@app.function(
    gpu="h200:2",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_h200_2():
    ssh_setup()

@app.function(
    gpu="h200:4",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_h200_4():
    ssh_setup()

@app.function(
    gpu="h200:8",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_h200_8():
    ssh_setup()

# B200 GPU configurations
@app.function(
    gpu="b200:1",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_b200_1():
    ssh_setup()

@app.function(
    gpu="b200:2",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_b200_2():
    ssh_setup()

@app.function(
    gpu="b200:4",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_b200_4():
    ssh_setup()

@app.function(
    gpu="b200:8",
    volumes={"/dev/shm/models": llama4_maverick_fp8, MODEL_DIR: MODEL_STORE_VOLUME},
    timeout=86400,
)
def sglang_b200_8():
    ssh_setup()

if __name__ == "__main__":
    show_help()
