from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_ssh_only_image_healthcheck_probes_sshd_not_llama_server():
    dockerfile = (ROOT / "Dockerfile.mx-llamacpp-ssh").read_text(encoding="utf-8")
    healthcheck = "\n".join(
        line.rstrip("\\").strip()
        for line in dockerfile.splitlines()
        if "HEALTHCHECK" in line or "socket.create_connection" in line
    )

    assert "HEALTHCHECK" in healthcheck, "must override the inherited llama-server healthcheck"
    assert "socket.create_connection(('127.0.0.1', 22), timeout=2)" in healthcheck
    assert "localhost:8080" not in healthcheck
