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
#               dependencies plus the [ui] extra, so `docker compose up` can
#               run the web UI), exported with hashes and installed
#               with --no-deps (the export is the complete closure), then the
#               project's own wheel, built from this source with the pinned
#               backend and no build isolation.
RUN python -m venv /opt/tools && python -m venv /opt/venv
RUN /opt/tools/bin/pip install --no-cache-dir --require-hashes -r requirements/uv.txt -r requirements/build-backend.txt
RUN /opt/tools/bin/uv export --frozen --no-emit-project --extra ui --format requirements-txt -o /tmp/requirements.txt
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --require-hashes --no-deps -r /tmp/requirements.txt
RUN /opt/tools/bin/pip wheel --no-cache-dir --no-deps --no-build-isolation --wheel-dir /tmp/wheels . && pip install --no-cache-dir --no-deps /tmp/wheels/*.whl && pip check

# bun, the JavaScript runtime the Reflex UI builds its frontend with. Pinned by
# version and SHA-256 (docker/fetch_bun.py). Without it Reflex would fetch bun
# at container start through `curl | bash`, which needs curl and unzip (not in
# the slim image) and runs an unpinned script.
FROM python:3.11-slim@sha256:2c285c669cc837aa3bcf1af23ea1932b7b5214f9c9d3aad22417446ad91cb4fb AS bun
ARG TARGETARCH
COPY docker/fetch_bun.py /tmp/fetch_bun.py
RUN python /tmp/fetch_bun.py "${TARGETARCH:-amd64}" /opt/bun/bin

FROM python:3.11-slim@sha256:2c285c669cc837aa3bcf1af23ea1932b7b5214f9c9d3aad22417446ad91cb4fb
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=bun /opt/bun/bin/bun /opt/bun/bin/bun
ENV PATH="/opt/venv/bin:/opt/bun/bin:$PATH"
# No second copy of the source under /app: the package is installed in
# /opt/venv, and a copy here shadowed it for `python -c` / `python -m` run from
# this directory (root-owned, so Reflex could not write its build output).
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 REFLEX_USE_SYSTEM_BUN=1
# `backprop ui` runs Reflex from the installed package directory, and Reflex
# writes there: build output (.web/, .states/, reflex.lock/), upload space
# (uploaded_files/) and two project files it insists on (.gitignore,
# requirements.txt). The package stays root-owned and read-only; only these
# paths belong to appuser.
RUN useradd -m -r appuser \
 && PKG="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/backpropagate" \
 && cd "$PKG" \
 && mkdir -p .web .states reflex.lock uploaded_files \
 && touch .gitignore requirements.txt \
 && chown appuser .web .states reflex.lock uploaded_files .gitignore requirements.txt
USER appuser

# Healthcheck exercises the entrypoint shim + minimal import tree.
# Provides 'docker ps' health signal for downstream orchestrators
# (Kubernetes, Docker Swarm, ECS) consuming this image as a base or
# running it as a long-lived training container.
HEALTHCHECK --interval=30s --timeout=15s --start-period=10s --retries=3 \
    CMD backpropagate --version >/dev/null 2>&1 || exit 1

ENTRYPOINT ["backpropagate"]
CMD ["--help"]
