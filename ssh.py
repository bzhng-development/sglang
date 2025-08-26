import modal
import subprocess
import time



sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:dev", add_python="3.12")
    .run_commands("apt-get update && apt-get install -y git clang openssh-server")
    .run_commands("mkdir -p /run/sshd /root/.ssh && chmod 700 /root/.ssh")
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
    .run_commands("git clone https://github.com/bzhng-development/sglang.git", force_build=False)
    .pip_install("uv")
    .run_commands("cd sglang && uv pip install -e 'python[all]' --system", force_build=False)
)

llama4_maverick_fp8 = modal.Volume.from_name("llama-4-maverick-17b-fp8")

app = modal.App("sglang-h200", image=sglang_image)

# A100 GPU variations
@app.function(gpu="a100:8", volumes={"/dev/shm/models": llama4_maverick_fp8}, timeout=86400)
def sglang_a100_8():
    # Start sshd in the background
    subprocess.Popen(["/usr/sbin/sshd", "-D", "-e"])
    # Expose port 22 to the public Internet as a raw TCP socket; SSH provides encryption.
    with modal.forward(port=22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print(f"ssh -o StrictHostKeyChecking=no -p {port} root@{host}")
        # Keep the container alive for 24 hours
        time.sleep(86400)

@app.function(gpu="a100:4", volumes={"/dev/shm/models": llama4_maverick_fp8}, timeout=86400)
def sglang_a100_4():
    # Start sshd in the background
    subprocess.Popen(["/usr/sbin/sshd", "-D", "-e"])
    # Expose port 22 to the public Internet as a raw TCP socket; SSH provides encryption.
    with modal.forward(port=22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print(f"ssh -o StrictHostKeyChecking=no -p {port} root@{host}")
        # Keep the container alive for 24 hours
        time.sleep(86400)

@app.function(gpu="a100:1", volumes={"/dev/shm/models": llama4_maverick_fp8}, timeout=86400)
def sglang_a100_1():
    # Start sshd in the background
    subprocess.Popen(["/usr/sbin/sshd", "-D", "-e"])
    # Expose port 22 to the public Internet as a raw TCP socket; SSH provides encryption.
    with modal.forward(port=22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print(f"ssh -o StrictHostKeyChecking=no -p {port} root@{host}")
        # Keep the container alive for 24 hours
        time.sleep(86400)

# H100 GPU variations
@app.function(gpu="h100:8", volumes={"/dev/shm/models": llama4_maverick_fp8}, timeout=86400)
def sglang_h100_8():
    # Start sshd in the background
    subprocess.Popen(["/usr/sbin/sshd", "-D", "-e"])
    # Expose port 22 to the public Internet as a raw TCP socket; SSH provides encryption.
    with modal.forward(port=22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print(f"ssh -o StrictHostKeyChecking=no -p {port} root@{host}")
        # Keep the container alive for 24 hours
        time.sleep(86400)

@app.function(gpu="h100:4", volumes={"/dev/shm/models": llama4_maverick_fp8}, timeout=86400)
def sglang_h100_4():
    # Start sshd in the background
    subprocess.Popen(["/usr/sbin/sshd", "-D", "-e"])
    # Expose port 22 to the public Internet as a raw TCP socket; SSH provides encryption.
    with modal.forward(port=22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print(f"ssh -o StrictHostKeyChecking=no -p {port} root@{host}")
        # Keep the container alive for 24 hours
        time.sleep(86400)

@app.function(gpu="h100:1", volumes={"/dev/shm/models": llama4_maverick_fp8}, timeout=86400)
def sglang_h100_1():
    # Start sshd in the background
    subprocess.Popen(["/usr/sbin/sshd", "-D", "-e"])
    # Expose port 22 to the public Internet as a raw TCP socket; SSH provides encryption.
    with modal.forward(port=22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print(f"ssh -o StrictHostKeyChecking=no -p {port} root@{host}")
        # Keep the container alive for 24 hours
        time.sleep(86400)

# H200 GPU variations
@app.function(gpu="h200:8", volumes={"/dev/shm/models": llama4_maverick_fp8}, timeout=86400)
def sglang_h200_8():
    # Start sshd in the background
    subprocess.Popen(["/usr/sbin/sshd", "-D", "-e"])
    # Expose port 22 to the public Internet as a raw TCP socket; SSH provides encryption.
    with modal.forward(port=22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print(f"ssh -o StrictHostKeyChecking=no -p {port} root@{host}")
        # Keep the container alive for 24 hours
        time.sleep(86400)

@app.function(gpu="h200:4", volumes={"/dev/shm/models": llama4_maverick_fp8}, timeout=86400)
def sglang_h200_4():
    # Start sshd in the background
    subprocess.Popen(["/usr/sbin/sshd", "-D", "-e"])
    # Expose port 22 to the public Internet as a raw TCP socket; SSH provides encryption.
    with modal.forward(port=22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print(f"ssh -o StrictHostKeyChecking=no -p {port} root@{host}")
        # Keep the container alive for 24 hours
        time.sleep(86400)

@app.function(gpu="h200:1", volumes={"/dev/shm/models": llama4_maverick_fp8}, timeout=86400)
def sglang_h200_1():
    # Start sshd in the background
    subprocess.Popen(["/usr/sbin/sshd", "-D", "-e"])
    # Expose port 22 to the public Internet as a raw TCP socket; SSH provides encryption.
    with modal.forward(port=22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print(f"ssh -o StrictHostKeyChecking=no -p {port} root@{host}")
        # Keep the container alive for 2 hours
        time.sleep(7200)

# B200 GPU variations
@app.function(gpu="b200:8", volumes={"/dev/shm/models": llama4_maverick_fp8}, timeout=86400)
def sglang_b200_8():
    # Start sshd in the background
    subprocess.Popen(["/usr/sbin/sshd", "-D", "-e"])
    # Expose port 22 to the public Internet as a raw TCP socket; SSH provides encryption.
    with modal.forward(port=22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print(f"ssh -o StrictHostKeyChecking=no -p {port} root@{host}")
        # Keep the container alive for 24 hours
        time.sleep(86400)

@app.function(gpu="b200:4", volumes={"/dev/shm/models": llama4_maverick_fp8}, timeout = 86400) 
def sglang_b200_4():
    # Start sshd in the background
    subprocess.Popen(["/usr/sbin/sshd", "-D", "-e"])
    # Expose port 22 to the public Internet as a raw TCP socket; SSH provides encryption.
    with modal.forward(port=22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print(f"ssh -o StrictHostKeyChecking=no -p {port} root@{host}")
        # Keep the container alive for 24 hours
        time.sleep(86400)

@app.function(gpu="b200:1", volumes={"/dev/shm/models": llama4_maverick_fp8}, timeout=86400)
def sglang_b200_1():
    # Start sshd in the background
    subprocess.Popen(["/usr/sbin/sshd", "-D", "-e"])
    # Expose port 22 to the public Internet as a raw TCP socket; SSH provides encryption.
    with modal.forward(port=22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print(f"ssh -o StrictHostKeyChecking=no -p {port} root@{host}")
        # Keep the container alive for 24 hours
        time.sleep(86400)