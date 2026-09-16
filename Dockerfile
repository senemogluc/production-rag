# API Dockerfile. NOT part of docker-compose.yml's default stack: the API runs on the host so
# it can use the GPU directly (see docs/v4-notes.md for why GPU passthrough wasn't wired up).
# Builds without GPU passthrough, falling back to CPU inference (llm.py already handles this via
# torch.cuda.is_available()); mainly useful for a future GPU-enabled cloud deployment. Note the
# image still pulls the CUDA-enabled torch wheel pinned in pyproject.toml/uv.lock, so it's a
# large image even though CUDA never activates without --gpus passthrough.

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS base

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY . .
RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "python", "serve.py"]
