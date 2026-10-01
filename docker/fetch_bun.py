"""Download a pinned bun release, verify its SHA-256, and install the binary.

Used by the Dockerfile so the image carries the JavaScript runtime the Reflex
web UI needs. Without it, Reflex fetches bun at container start through
``curl | bash``, which needs ``curl`` and ``unzip`` (absent from the slim base
image) and runs an unpinned script. Standard library only.

Usage: python fetch_bun.py <docker TARGETARCH> <dest dir>
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import urllib.request
import zipfile

VERSION = "1.3.13"
# From https://github.com/oven-sh/bun/releases/download/bun-v1.3.13/SHASUMS256.txt
ASSETS = {
    "amd64": ("bun-linux-x64", "79c0771fa8b92c33aae41e15a0e0d307ea99d0e2f00317c71c6c53237a78e25a"),
    "arm64": ("bun-linux-aarch64", "70bae41b3908b0a120e1e58c5c8af30e74afae3b8d11b0d3fdd8e787ddfb4b22"),
}


def main(arch: str, dest: str) -> int:
    if arch not in ASSETS:
        print(f"fetch_bun: no pinned bun build for architecture {arch!r}", file=sys.stderr)
        return 1
    name, expected = ASSETS[arch]
    url = f"https://github.com/oven-sh/bun/releases/download/bun-v{VERSION}/{name}.zip"
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 - fixed https URL
        data = resp.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        print(f"fetch_bun: SHA-256 mismatch for {name}.zip: {actual} != {expected}", file=sys.stderr)
        return 1
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        binary = archive.read(f"{name}/bun")
    os.makedirs(dest, exist_ok=True)
    target = os.path.join(dest, "bun")
    with open(target, "wb") as fh:
        fh.write(binary)
    os.chmod(target, 0o755)  # noqa: S103 - an executable on PATH, read-only to others
    print(f"fetch_bun: installed bun {VERSION} ({name}) at {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
