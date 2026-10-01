"""Dead-man's switch: delete a RunPod pod at a deadline, whatever happens to the session.

Start it right after creating a pod, detached from the session that created
it, so an interrupted session can never leave a pod billing:

    # Windows (PowerShell / Git Bash)
    powershell -NoProfile -Command "Start-Process -WindowStyle Hidden -FilePath python \
        -ArgumentList 'scripts/runpod/deadman.py','<POD_ID>','3.25','deadman.log'"
    # Linux / macOS
    nohup python scripts/runpod/deadman.py <POD_ID> 3.25 deadman.log &

Arguments: pod id, hours until deletion, log file. Needs RUNPOD_API_KEY.
A 404 at the deadline means the pod was already deleted, which is fine.

The timer runs on the machine that started it. If that machine sleeps or
hibernates past the deadline, the delete fires on wake (the loop compares
wall-clock time) and the pod bills until then. Keep the machine awake while a
pod is up, or run `scripts/runpod/pod.sh list` after a sleep.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time


def main() -> int:
    pod, hours, log = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    deadline = time.time() + hours * 3600
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"armed pod={pod} deadline={time.ctime(deadline)}\n")
    while time.time() < deadline:
        time.sleep(60)
    for _ in range(10):
        # curl, not urllib: Cloudflare blocks Python's default client (error 1010).
        r = subprocess.run(
            ["curl", "-s", "-o", os.devnull, "-w", "%{http_code}", "-X", "DELETE",
             f"https://rest.runpod.io/v1/pods/{pod}",
             "-H", f"Authorization: Bearer {os.environ['RUNPOD_API_KEY']}"],
            capture_output=True, text=True, check=False,
        )
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"{time.ctime()} DELETE -> {r.stdout}\n")
        if r.stdout.strip() in ("200", "204", "404"):
            return 0
        time.sleep(120)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
