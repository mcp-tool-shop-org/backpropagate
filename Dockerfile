# Base image pinned to python:3.11.15-slim by multi-arch manifest-list digest.
# Refresh digest with:
#   curl -s "https://hub.docker.com/v2/repositories/library/python/tags/3.11-slim" | python -c "import json,sys; print(json.load(sys.stdin).get('digest',''))"
# OR locally with: docker manifest inspect python:3.11-slim
# When updating, bump BOTH FROM lines together — Dependabot's docker ecosystem
# (added in .github/dependabot.yml) will open a PR per digest change.
#
# Python version note: pyproject.toml declares `requires-python = ">=3.10"`
# and the PyPI wheel is `py3-none-any` (works on 3.10, 3.11, 3.12, 3.13). This
# image pins to 3.11 as the shipped runtime — it is one of several supported
# Pythons, not the minimum. Operators on 3.10 / 3.12 / 3.13 should install
# from PyPI directly into their own interpreter rather than this image.
FROM python:3.11-slim@sha256:2c285c669cc837aa3bcf1af23ea1932b7b5214f9c9d3aad22417446ad91cb4fb AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY requirements/uv.txt requirements/build-backend.txt requirements/
COPY backpropagate/ backpropagate/

# Every package in the image is installed with its hash checked (OpenSSF
# Scorecard Pinned-Dependencies, and a reproducible image).
#   /opt/tools  uv + the hatchling build backend, from hash-pinned files. It
#               stays in this stage; only /opt/venv is copied to the final image.
#   /opt/venv   the runtime venv: the dependency closure of uv.lock (core
#               dependencies, no extras), exported with hashes and installed
#               with --no-deps (the export is the complete closure), then the
#               project's own wheel, built from this source with the pinned
#               backend and no build isolation.
RUN python -m venv /opt/tools && python -m venv /opt/venv
RUN /opt/tools/bin/pip install --no-cache-dir --require-hashes -r requirements/uv.txt -r requirements/build-backend.txt
RUN /opt/tools/bin/uv export --frozen --no-emit-project --format requirements-txt -o /tmp/requirements.txt
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --require-hashes --no-deps -r /tmp/requirements.txt
RUN /opt/tools/bin/pip wheel --no-cache-dir --no-deps --no-build-isolation --wheel-dir /tmp/wheels . && pip install --no-cache-dir --no-deps /tmp/wheels/*.whl && pip check

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
