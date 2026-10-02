# Handoff: backpropagate UI v2 (to Kimi)

Written 2026-10-01 by the lead session. Read all of it, then open the UI and the
reference design yourself before writing code.

## Your role

You own the backpropagate web UI end to end for 1.8.2: **function and visual design.**

| Who | Owns |
|---|---|
| **The Director** | Judges the visuals. He reviews your screenshots before any UI PR merges. |
| **The lead (Claude)** | Reviews correctness, security, CI and tests. Writes the CHANGELOG and handbook text. Makes **no** visual design calls. |

**Why this handoff exists.** The Director ruled that the UI is the product's face, and on the Microsoft Store it *is* the product. Earlier handoffs only checked plumbing (ports, auth, packaging) and never asked anyone to use the UI. The result was a UI with broken images, three "coming soon" primary actions, fake status values and an unpolished layout. The Store submission is on hold until the UI is good.

## What the UI looks like today

Screenshots taken 2026-10-01 at 1920×1080, from main plus #279 (#279 fixes the broken images): `E:/AI/bp-shots/screens/draft2/01..06-*-dark.png`. Look at all six.

**Function:**
- **"Coming soon":** Single run's **Start training**, Multi-run's **Start multi-run** and Export's **Export** are all marked "coming soon". None of them does anything.
- **Runs page:** always empty, because `Trainer.save` wiped `run_history.json` (fixed in #278).
- **Runs page copy:** tells users to "click Start in the UI".
- **Models page:**
  - shows a literal `<redacted-path>`;
  - may ignore `HF_HOME` (this rig's cache is `E:\AI-Models\hf-cache`, with many models, but only 3 were listed);
  - its description hard-codes `~/.cache/huggingface/hub/`.
- **Status panel (every page):** VRAM fixed at "0.0 / 16.0 GB" on a 32 GB card, 0 °C, loss "0 … 0".
- **Defaults copy** says "Qwen 2.5 7B on a 16 GB GPU", but the product is positioned 32 GB-first.
- **Missing compared with the CLI:**
  - presets (`config.py`)
  - method (SFT / ORPO / SimPO / KTO)
  - full fine-tuning mode
  - the VRAM estimator (`backprop estimate-vram`)

**Visual (your call how to fix; these are just the obvious ones):**
- **Layout:** the form uses about 40% of a 1920-wide screen and sits flush against the sidebar with no gutter. Section labels touch their inputs and spacing is cramped.
- **"Advanced" / "Filter / Dedup"** disclosures render as full-width teal bars that look like buttons.
- **Export:** the three format descriptions run together with no spacing.
- **Small icons:** nav icons and header controls (theme toggle, GitHub) are tiny.
- **Footer:** shows a literal `[KEY]`. The "Built with Reflex" badge covers the GitHub link.
- **Fonts:** Geist never loads. The Google Fonts stylesheet is blocked by our own CSP (`style-src 'self'`), and the Store build is offline anyway. **Self-host Geist** (OFL-licensed) in `backpropagate/assets/` and serve it locally.

## The reference design (the intended look; the code drifted from it)

A Claude Design pass in May 2026 produced the design system and mockups of every screen:
- **Mockups:** `E:/AI/dogfood-labs/swarms/swarm-1779335775-02be/stage-d/claude-design-out/prompt-1-reflex-ui/bundle.html`. Open it in Chrome; it holds the Shell, Train, Multi-Run, Export and Dataset surfaces.
- **Distilled tokens:** `design-digest.md` in the same folder ("Ocean Mist": dark and light tokens, radii, Geist / Geist Mono, focus ring).
- **The original prompts:** `.../stage-d/claude-design-prompt-1-REFLEX.md` and `claude-design-prompts.md`.

The tokens reached `backpropagate/ui_theme.py`; the layouts and polish did not. **Close that gap.**

**The Director's visual direction (2026-10-02, after reviewing P1).** He judged the P1 look dated and boxy, like a late-90s form, and asked for a modern app with curves, proper spacing and room to breathe. Where it conflicts with the old mockups, this direction wins. It applies to every phase:
- **Curves:** cards 12–16 px radius, inputs and selects 8–10 px, buttons pill-shaped or 10 px. No square boxes, and no hard 1 px grid lines between every field.
- **Spacing:** one scale (4/8/12/16/24/32/48) used everywhere:
  - 24–32 px gutter between the sidebar and the content;
  - 6–8 px from a label to its input;
  - 20–24 px between fields, 32–48 px between sections.
- **Layout:**
  - content grouped in rounded cards on a slightly different page background, with a soft shadow or subtle elevation;
  - a maximum content width;
  - two columns on wide screens, never a crowded form next to half an empty screen.
- **Type:** page title 28–32 px, section titles 16–18 px semibold, labels 13–14 px. Numbers formatted, e.g. loss to 3–4 decimals.
- **Icons:** nav and header icons 20 px, with comfortable hit areas.
- **Sidebar:** the active page shown as a rounded pill highlight.
- **Run progress:** its own card with:
  - a large step counter ("26 / 400");
  - a rounded progress bar;
  - the heartbeat and ETA;
  - the loss chart in a card.
- **Motion:** short, subtle transitions on hover, focus and state changes.
- **References:** current dashboards such as Linear, Vercel and Raycast. Keep the Ocean Mist palette and Geist if they fit this direction.
- **Light and dark** must both look finished.
- **You may improve on the mockups.** You own the design.
- **Keep:** the token system, WCAG AA contrast and the visible focus ring.
- **Make it fit real screens:** 1366×768 (the Store's minimum screenshot size) through 2560+ wide, and a narrow window.

## What to build

The functional design, architecture and research are in **`docs/design-ui-v2.md`** (same branch as this file). Read its research tables: they record what other fine-tuning GUIs got wrong. You may change the architecture if you find something better.

**Requirements you must keep:**
1. **Training never runs inside the UI server process.** Run it as a subprocess that writes progress to files in its run directory. The UI reads those files, and no pipe needs draining.
2. **Stop = "Stop and save checkpoint".**
   - It's cooperative and takes effect at the next step, through a control file and an HF `TrainerControl` callback. Never a POSIX signal: SIGABRT is a hard kill on Windows.
   - After a grace period, kill the whole process tree: a Windows Job Object with kill-on-close, POSIX `killpg`. The 1.8.1 `_kill_process_tree` and `_run_reflex` in `cli.py` show the existing patterns.
3. **State comes from disk** (PID + process start time + a job file). A page reload, or restarting `backprop ui`, reattaches or reports the real state.
4. **One job at a time,** and a second start is refused **on screen**.
5. **Start and stop travel over the existing authenticated, Origin-checked WebSocket.**
6. **`trust_remote_code` is refused for remote clients** (`--share` / non-loopback `--host`) unless an explicit server flag allows it.
7. **Server-side caps** on steps, batch size, sequence length and upload size.
8. **GPU safety stops** (`gpu_safety.GPUMonitor`) go through the same stop path and show as events.
9. **Status panel shows real values:** `gpu_safety.get_gpu_status()` gives VRAM used and total, temperature and utilisation. When no GPU is present, say so instead of showing zeros.
10. **Progress UX, from the research:**
    - step/total from step 0, with named phases before step 1;
    - a heartbeat ("last step N s ago") that flags a stall;
    - an ETA only after warm-up, shown as a range;
    - raw loss under a smoothed (debiased EMA) line;
    - notifications only on finished, failed, stopped or blocked;
    - on failure, show the last N log lines and the error code.

## Phases and acceptance gates

Each phase is its own PR. **A PR is not ready until every gate item is met.**

| Phase | Scope |
|---|---|
| P1 | Job runner + Single run end to end + real status panel + the visual foundation (layout grid, spacing scale, self-hosted fonts, header/footer fixes) |
| P2 | Export and Multi-run through the same job runner; Runs and Run-detail live views; Models page fixes |
| P3 | Full visual pass on every page + CLI parity (presets, method, mode, VRAM estimate inline: fits / tight / won't fit) |

**Gate for every phase:**
1. **A real browser test of what the phase adds, run once before the PR.** Playwright with `channel="chrome"`; a scratch venv with Playwright exists at `E:/AI/bp-shots/.shotvenv`.
   - **P1:** start a real 20-step run from the UI, watch steps advance, press Stop, assert a loadable checkpoint and a `run_history.json` entry, reload mid-run and assert it reattaches.
   - **P2:** export a GGUF from the UI, and run a 2-run multi-run from the UI.
   - **P3:** each new control (preset, method, mode) starts a run with the setting it shows; the VRAM estimate matches `backprop estimate-vram`.
   - **These touch the GPU:** tell the lead first, and the lead confirms with the Director. The VRAM watchdog must be up (`pwsh -NoProfile -File E:\AI\training\_watchdog_start.ps1`).
2. **No "coming soon"** text anywhere in the UI.
3. **Screenshots**, saved to `E:/AI/bp-shots/review/<phase>/` and listed in the PR body:
   - one per page the phase changed, at 1920×1080, in dark and light;
   - for the training and run pages, also running, stopped and finished.
   - **Look at them yourself before sending them.** The Director reviews them before merge.
   - The 1366×768 set is taken once, from the installed Store build after P3; it doubles as the Store listing images.
4. **Accessibility, once, in P3:** WCAG AA text contrast, and a visible focus ring on every interactive element. Tab through each page once.
5. **CI green on GitHub,** confirmed via REST (`gh api repos/<r>/commits/<sha>/check-runs`), after `atlas check` on the final commit.

**Don't over-test (Director, 2026-10-02):**
- Don't run the full local suite (7,300+ tests) before a PR, or after every change. CI runs it on every push. Locally, run only the test files that cover what you changed.
- Run each GPU or browser test once, when its feature is done, not after every edit.
- The lead doesn't re-run your tests: it reads the diff, looks at the screenshots and confirms CI.

## Open fixes your work depends on

- **#277:** `BACKPROPAGATE_UI__OUTPUT_DIR` crashed every command.
- **#278:** save no longer wipes the output folder.
- **#279:** images and icons. The rate limiter now counts only rejected auth attempts, and the workdir mirrors `assets/`.
- **A W&B fix in progress:** `report_to="auto"` crashed training when wandb was installed but not logged in. **Until it lands, run training with `WANDB_MODE=disabled`.**

Branch from `main` after these merge. Or branch now and rebase.

## Repo rules

- **Worktree:** work in **your own git worktree**. Other sessions use the main checkout `E:/AI/backpropagate`, and a separate session is working in it now.
  - The rig venv `E:/AI/backpropagate/.venv` is an editable install of the main checkout, so set `PYTHONPATH=<your worktree>` when running.
  - Its package metadata says 1.7.0, so the UI header shows "v1.7.0" on this rig. That's a rig artifact, not a bug to fix.
- **Bot PRs:** open PRs as the `mcp-tool-shop-bot` GitHub App (token helper `C:\Users\mikey\.config\mcp-tool-shop-bot\mint_token.py`, run with the repo venv). Never print the token.
  - **Never approve a PR.** The Director approves using Run lines the lead gives him.
  - **When GraphQL is rate-limited** (the quota is shared across the rig), use REST (`gh api`).
- **Lead-written docs:** don't edit README, CHANGELOG, handbook or SECURITY. List the changes you need in the PR body, and the lead writes them.
- **Atlas:** after code changes, run `npx --yes @dogfood-lab/atlas@1.24.0 check`. If it goes red after an intended change, run `map` and commit `atlas/`. Never hand-edit `atlas/`.
- **Drift check:** `scripts/check_doc_drift.py` must pass. New flags and env vars need handbook rows; bridge them in `scripts/doc-drift-allow.toml` with a comment, and the lead writes the rows.
- **CRLF working copies:** when patching from a shell, preserve each file's line endings.
- **Known rig-only test failures:** `TestStageCHfTokenCap` and the `HF_TOKEN`-dependent export test (run with `HF_TOKEN` unset), plus FP8 / full-FT GPU tests and two LR tests that are flaky under xdist.
- **Subagents** run on Sonnet. **No Ollama Cloud.**

## After P3

1. The Store track resumes: re-pack the MSIX (`scripts/build_msix.py`) from main.
2. Sideload it. The Director runs the cert trust commands.
3. Take the **Store listing screenshots from the installed Store build**, at 1366×768 or larger.
4. Submit with the publishing hold already decided: the Director publishes after checking the certified build.

## Reporting, per PR

- What changed, **visually and functionally**.
- The screenshot folder.
- Browser-test evidence: what the test drove, the step count reached, the checkpoint found.
- CI REST-verified.
- The doc changes the lead needs to write.
