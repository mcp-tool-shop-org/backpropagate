# Design: backpropagate UI v2 — a UI that actually trains

Status: draft for Director review, 2026-10-01. Target: 1.8.2. Blocks the Microsoft
Store submission (the Store tile opens this UI, so on the Store it is the product).

## Where the UI is today (observed in a real browser, 2026-10-01)

| Page | Primary action | State |
|---|---|---|
| Single run | Start training | "coming soon": no training happens |
| Multi-run | Start multi-run | "coming soon" |
| Export | Export | "coming soon" |
| Dataset | Upload, detect, preview, filter | works |
| Runs | List past runs | works, but empty: `Trainer.save` wiped `run_history.json` (fix in #278) |
| Models | List / delete HF cache | works, but shows a literal `<redacted-path>`, and may ignore `HF_HOME` |

**Status panel (every page): placeholders.**
- VRAM is fixed at "0.0 / 16.0 GB", temperature at 0 °C, and loss at "0 … 0".

**Visual problems:**
- Images broken (fixed in #279).
- The form uses about 40% of the width, flush against the sidebar.
- "Advanced" renders as a full-width teal bar.
- The Export format descriptions run together.
- Tiny nav and header icons.
- A literal `[KEY]` in the footer.
- The "Built with Reflex" badge covers the GitHub link.

**Missing compared with the CLI:**
- presets
- method (SFT / ORPO / SimPO / KTO)
- full fine-tuning mode
- the VRAM estimate

## Acceptance (the bar for "done", per phase)

1. Every primary action does its job end to end in a real browser. A Playwright test (channel `chrome`) starts a real short training from the UI, sees steps advance, presses Stop, and finds a loadable checkpoint and a `run_history.json` entry.
2. No "coming soon" anywhere.
3. The status panel shows real values: device VRAM total and used, temperature, step/total, loss.
4. A screenshot set of every page, idle and mid-run, at 1920×1080, reviewed by the lead **by looking at it** and then by the Director. Checks and test counts are not the review.
5. Reload the page mid-run → it reattaches. Restart `backprop ui` mid-run → it reattaches, or reports the run's real state.

## Research grounding

Three research reports, 2026-10-01: prior art, progress UX, and job-control safety. Each finding below carries its source and what it changes in this design.

**Prior art:**

| Source | Finding | Implication for us |
|---|---|---|
| LLaMA-Factory `webui/runner.py`, issue #10180 | Training runs as a subprocess. A pipe the UI had to drain stalled training overnight once the browser left. | Run training as a **subprocess writing to files in its run directory**, never a pipe the UI must drain. |
| H2O LLM Studio `get_experiments_status` | Status comes from the PID plus a child-written state file; a dead PID still marked "running" becomes "failed". | **Derive state from disk** (PID + process start time + `job.json`), so reload and UI restart are harmless. |
| LLaMA-Factory `Engine.resume`, against kohya #2749 and #2782 | Rebuilding the page from server state on load works; per-browser button state goes stale after a reload. | On page load, **rebuild the page from server state**. |
| LLaMA-Factory `callbacks.py` (SIGABRT handler); oobabooga `WANT_INTERRUPT` | Graceful stop happens at a step boundary. SIGABRT is a hard kill on Windows. | **Cooperative stop via a control file** checked at each step. Never a POSIX signal. |
| kohya `kill_command`, #1216 | Stop is a hard kill that loses everything since the last checkpoint. | Stop must **save**, not just kill. |
| kohya `execute_command` | A second start is refused with a message only in the log. | Refuse a second start **visibly in the UI**. |
| LlamaFactory #3978 | Stop failed to reach all the processes in a multi-GPU run. | Kill the **whole process tree** as the fallback. |
| oobabooga #2956 | Training in the server process mutated the shared model. | **Never train inside the UI server process.** |
| kohya `class_command_executor` | A ring buffer of recent output makes failures explainable. | On failure, show the **last N log lines** and the error code. |

**Job-control safety:**

| Source | Finding | Implication for us |
|---|---|---|
| HF `TrainerControl` docs | A callback stops cleanly at a step boundary by setting both `should_training_stop` and `should_save`. | Stop sets both flags at the next step. |
| Kubernetes pod lifecycle | Stop is graceful first, then forced after a grace period. | Use a grace period, then kill the tree. |
| Microsoft Learn, Job Objects | A Job Object with kill-on-close cleans up grandchildren, even if the parent crashes. | **Windows:** run the job in a Job Object with kill-on-close (DataLoader workers included). **POSIX:** `start_new_session` plus `killpg`. |
| OWASP WSTG (WebSockets); Jupyter Server security | Origin checks and authenticated channels for anything that starts code. | **Start and stop are events on the authenticated, Origin-checked WebSocket.** These checks already exist in 1.8.1: keep them. |
| Jupyter Server security (remote kernels count as code execution) | Exposing job start remotely amounts to remote code execution. | **`trust_remote_code` is refused for remote clients** (`--share` / non-loopback `--host`) unless an explicit server flag allows it. |
| — | Unbounded inputs let one request exhaust the machine. | Server-side caps on steps, batch size, sequence length and upload size. |

**Progress UX:**

| Source | Finding | Implication for us |
|---|---|---|
| Myers 1985 (CHI); Nielsen's response-time limits | A percent-done indicator is wanted for anything past about 10 seconds. | Show **step/total and a bar from step 0**, plus named phases before step 1 (loading model, tokenizing, compiling, training, saving). |
| Nah 2004 | Visible activity raises how long people will tolerate waiting. | A **heartbeat**: "last step 2 s ago". If no step lands for N seconds, say "no progress for N s" instead of freezing. |
| Harrison et al. 2007 (UIST) | Smooth, frequent updates make a wait feel shorter. | Update on every logged step. |
| Conrad et al. 2010 | Discouraging early progress readings hurt. | **No ETA during warm-up**: show it after about 20 steps or 5%, from a trailing-window rate, rounded. |
| Kay et al. 2016 (CHI) | Showing uncertainty in a predicted time improves people's estimates. | Show the ETA **as a range** ("35–50 min"). |
| TensorBoard EMA practice (no HCI study exists) | Smoothing is the standard way to read noisy loss. | **Raw loss faint, debiased EMA bold**, with a toggle and a "smoothed" label. |
| Adamczyk & Bailey 2004; Mark et al. 2008 | Interruptions cost most when they land mid-task. | **Notify only on terminal or blocked states** (finished, failed, stopped, thermal pause that won't clear). Everything else goes to the in-page event log. |
| NN/g confirmation-dialog guidance | Confirmations belong only to destructive actions. | **Stop = "Stop and save checkpoint"**, one click, no confirmation, then a "Stopping… finishing step, saving" state and a report of what was kept. Only a separate "Discard run" asks for confirmation. |

## Architecture

```
browser ──(authenticated WS: start / stop events)──► UI server (Reflex)
                                                       │  JobManager (one job at a time)
                                                       │   spawns, never trains in-process
                                                       ▼
                    python -m backpropagate train ... --ui-run-dir <run_dir>
                       ├─ UiEventCallback  → <run_dir>/events.jsonl  (step, loss, lr, it/s,
                       │                       VRAM used/total, temp, phase, checkpoint saved)
                       ├─ StopFileCallback ← <run_dir>/control.json  (stop → should_training_stop
                       │                       + should_save at the next step)
                       └─ stdout/stderr    → <run_dir>/output.log
   JobManager state on disk: <run_dir>/job.json (pid, process start time, run_id, state)
   UI tails events.jsonl (async background task, about 1 s) → status panel + run page
```

1. **The JobManager** is new, in `backpropagate/ui_jobs.py`.
   - **Start:** `start(spec)` validates the spec server-side, refuses if a job is running (and says so in the UI), runs a VRAM pre-check with the existing estimator, and spawns the CLI as a subprocess.
     - **Windows:** inside a Job Object with kill-on-close.
     - **POSIX:** in its own session.
   - **Stop:** writes `control.json`, then waits a grace period of `max(60 s, 3 × last step time + measured save time)`, then kills the process tree. The cleanup helpers added in 1.8.1 can be reused.
   - **Status:** comes from disk. "Running" means the PID is alive and its start time matches `job.json`; anything else is finished, failed or stopped, decided by the last event.
2. **Child side:** two `TrainerCallback`s, wired by a new `--ui-run-dir` flag (hidden from `--help`) so CLI runs are unchanged.
   - **Event writer:** GPU samples come from `gpu_safety.get_gpu_status`.
   - **Stop-file poller.**
   - **The same path for safety stops:** a thermal pause or stop from `GPUMonitor` uses the stop path and logs an event, so safety stops look like user stops.
3. **Transport:** the UI reads files. No browser connection keeps the job alive, and none is needed to watch it.
4. **Pages:**
   - **Single run gains:** a preset picker, method, QLoRA/LoRA/full mode, and an inline VRAM estimate (fits / tight / won't fit).
   - **Multi-run and Export:** use the same JobManager (export is a short job with the same lifecycle).
   - **Runs:** shows the live run on top. **Run detail:** shows the event log and the loss chart.
5. **Status panel:** real device name, VRAM used/total, temperature, utilisation, run state, step, ETA range and loss. When no GPU is present it says so, rather than showing zeros.

## Visual pass (phase 3)

- **Layout:** content width and gutters, form spacing, and section headers that don't touch inputs.
- **Controls:** "Advanced" as a real disclosure; a responsive two-column layout at ≥1600 px; readable nav and header icons.
- **Text bugs:** the `[KEY]` and `<redacted-path>` literals.
- **Badge:** hide the Reflex badge or move it off the footer links.
- **Defaults:** copy fixes, e.g. the default hint should match the 32 GB-first positioning and the detected card.
- **Themes:** check light and dark mode, both reviewed from screenshots.

## Phases (each gated by the acceptance section above)

| Phase | Scope | Gate |
|---|---|---|
| P1 | JobManager + child callbacks + Single run end to end + real status panel | Playwright: real 20-step run from the UI, Stop saves a checkpoint, reload reattaches; screenshots reviewed |
| P2 | Export and Multi-run through the JobManager; Runs/Run detail live view | Playwright: export a GGUF from the UI; a 2-run multi-run; screenshots reviewed |
| P3 | Visual pass + CLI parity (presets, method, mode, VRAM estimate) | Full screenshot set (idle + mid-run, light + dark) reviewed by the lead, then the Director |

Then the Store package is re-packed and its screenshots taken **from the installed Store build**.

## Out of scope for 1.8.2

- multi-GPU
- job queues
- several concurrent jobs
- a tray app
- UI-driven Unsloth install

## Known dependencies

- **#278** (save no longer wipes the output folder): needed so Runs shows anything.
- **#279** (icons and rate limiter).
- **#277** (`BACKPROPAGATE_UI__OUTPUT_DIR` crash).
- **The report_to fix** (W&B crash), in progress.
