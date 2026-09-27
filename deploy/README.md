# Deploying the gfx906 llama.cpp image and the containers around it

The files at the repository root (`Dockerfile.mx-llamacpp-ssh`,
`docker-compose.yml`, `entrypoint.sh`, `fixup-gfx906.Dockerfile`,
`gfx906-rocm10.Dockerfile`) are a snapshot of what `/mnt/user/appdata/mx-llamacpp`
holds on the box today: the ssh container, and the two Dockerfiles that build the
llama.cpp image it runs on.

The directories here are the next versions. They overlap deliberately, so deploy
one of them into a given directory, not two.

## Which one

| directory | container | base image | what it changes |
|---|---|---|---|
| `mx-llamacpp-logs-only/` | `mx-llamacpp-ssh` (2222) | `zenth815/mx-llama-rocm10-gfx906:latest` | `sshd -D -e` and the `serve` wrapper put sshd and hand-run `llama-server` output on the container console, plus a log volume readable from the host. Pulls the base from Docker Hub and its compose drops the `name:` and `image:` pins, so the built image is renamed from `mx-llamacpp-mx-llamacpp-ssh:latest`. |
| `self-build/` | `mx-llamacpp-ssh` (2222) | `mx-llama-rocm10-gfx906:local` | Compiles llama.cpp for gfx906 from this repository instead of pulling the Hub image. Stage 2 keeps the box's current `FROM mx-llama-rocm10-gfx906:local` and adds the `serve` wrapper; the ROCm and HF environment moves into stage 1. |
| `mx-llamacpp-gui/` | `mx-llamacpp-gui` (8643) | `llama-swap-rescan:local` | llama-swap becomes PID 1 with sshd alongside, so the proxy log and every child `llama-server` log reach `docker logs` and the GUI on 8643. This compose declares the healthcheck the ssh image never had, which is where the `(unhealthy)` status comes from. |

`serve.sh` exists because a `llama-server` started by hand inside tmux writes to
the container's pty, and `docker logs` reads only PID 1's stdout/stderr. The
wrapper tees to a log file and to `/proc/1/fd/1`.

## Deploying `mx-llamacpp-logs-only`

```bash
cd /mnt/user/appdata/mx-llamacpp
cp .../deploy/mx-llamacpp-logs-only/{Dockerfile.mx-llamacpp-ssh,entrypoint.sh,serve.sh,docker-compose.yml} .
mkdir -p logs
docker compose up -d --build
docker logs -f mx-llamacpp-ssh
```

Inside the container, use `serve` instead of `llama-server` so the child's output
follows the same path:

```bash
ssh -p 2222 root@192.168.20.5
serve -m /app/models/HFCache/hub/.../model.gguf --port 8000 -ngl 99
```

## Deploying `self-build`

Stage 1 builds the image the ssh layer sits on. The Dockerfile expects to live in
`.devops/`, which is where this repository keeps its Dockerfiles:

```bash
# 1. build llama.cpp for gfx906
cp .../deploy/self-build/Dockerfile.mx-llama-rocm10-gfx906 .devops/mx-llama-rocm10-gfx906.Dockerfile
docker build -f .devops/mx-llama-rocm10-gfx906.Dockerfile -t mx-llama-rocm10-gfx906:local .

# 2. rebuild the ssh layer (entrypoint.sh and serve.sh come from
#    mx-llamacpp-logs-only/ above)
cp .../deploy/self-build/Dockerfile.mx-llamacpp-ssh /mnt/user/appdata/mx-llamacpp/
cd /mnt/user/appdata/mx-llamacpp && docker compose up -d --build

# 3. rebuild llama-swap on top of it, from the llama-swap-rescan checkout
docker build -f docker/llama-swap-source.Containerfile -t llama-swap-rescan:local .
```

`deploy/self-build/README.md` has the reasoning behind the ROCm base, including
why it is the `Wizard815/TheRock-gfx906` tree rather than a stock ROCm tag.

## Deploying `mx-llamacpp-gui`

```bash
mkdir -p /mnt/user/appdata/mx-llamacpp-gui
cp .../deploy/mx-llamacpp-gui/* /mnt/user/appdata/mx-llamacpp-gui/
docker compose -f /mnt/user/appdata/mx-llamacpp-gui/docker-compose.yml up -d --build
```

Two things to know before starting it:

- It maps 2222, 8000, 8001 and 8002, the same ports as `mx-llamacpp-ssh`, so stop
  that container first or drop those mappings.
- The compose mounts `/mnt/user/appdata/mx-llamacpp/config.yaml` and
  `/mnt/user/appdata/mx-llamacpp/config.d`, so the two containers share one
  config. The `config.yaml` in this directory is the pre-tuning one
  (`healthCheckTimeout: 120`, `-c 262144`, `--fit off`); the tuned config lives in
  the `llama-swap-rescan` repository under `deploy/`.
