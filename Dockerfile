# Base image pinned to python:3.11.15-slim by multi-arch manifest-list digest.
# Refresh digest with:
#   curl -s "https://hub.docker.com/v2/repositories/library/python/tags/3.11-slim" | python -c "import json,sys; print(json.load(sys.stdin).get('digest',''))"
# OR locally with: docker manifest inspect python:3.11-slim
# When updating, bump BOTH FROM lines together — Dependabot's docker ecosystem
# (added in .github/dependabot.yml) will open a PR per digest change.
FROM python:3.11-slim@sha256:2c285c669cc837aa3bcf1af23ea1932b7b5214f9c9d3aad22417446ad91cb4fb AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md LICENSE ./
COPY backpropagate/ backpropagate/
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

FROM python:3.11-slim@sha256:2c285c669cc837aa3bcf1af23ea1932b7b5214f9c9d3aad22417446ad91cb4fb
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY --chown=root:root backpropagate/ backpropagate/
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN useradd -m -r appuser
USER appuser

# Healthcheck exercises the entrypoint shim + minimal import tree.
# Provides 'docker ps' health signal for downstream orchestrators
# (Kubernetes, Docker Swarm, ECS) consuming this image as a base or
# running it as a long-lived training container.
HEALTHCHECK --interval=30s --timeout=15s --start-period=10s --retries=3 \
    CMD backpropagate --version >/dev/null 2>&1 || exit 1

ENTRYPOINT ["backpropagate"]
CMD ["--help"]
