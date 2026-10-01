# Handoff: backpropagate on the Microsoft Store (MSIX)

Written 2026-10-01 by the lead session for the agent that designs and builds the
Store package. Read all of it before writing code.

## Goal

Publish backpropagate to the Microsoft Store as an MSIX app. The first Store
version is **1.8.2.0**. 1.8.1 is in flight: PRs #266, #267 and #268, plus a
Ctrl+C fix for `backprop ui` on Windows. The Store package needs product changes
(items 1 and 2 below) that ship in 1.8.2.

## Decided by the Director (2026-10-01)

| Question | Decision |
|---|---|
| Torch build | **CUDA** (NVIDIA GPUs train; CPU fallback elsewhere). |
| Start-menu tile | **Console + browser**: open a console showing the `backprop ui` banner and open the default browser at the banner's `?token=` URL. Closing the console stops the UI. **Design is yours.** |
| Privacy policy | **A handbook page.** The lead writes it (public docs are lead-written). The listing links to it. |
| Store name | **backpropagate** (product 9MVXLZVL3TMT). The Director also reserved **"backprop"**; keep it reserved (possible future CPU-only edition, ~1 GB, no CUDA torch). Do not submit under it now. |
| Unsloth | Director asked for "both options at install". **MSIX has no install-time options.** Open question below. |

## Store facts (checked 2026-10-01)

- **Signing:** the Store re-signs MSIX/AppX packages with a Microsoft certificate after certification. No CA certificate is needed. Self-sign only for local sideload tests.
- **Size:** at most 25 GB per `.msix` or `.msixbundle`.
- **Version:** four parts. The fourth must be `0` (reserved for the Store), and the first cannot be 0.
- **Restricted capability:** desktop apps need `runFullTrust` (the `rescap` namespace). The submission asks why each restricted capability is needed. Say: GPU training through PyTorch/CUDA, running a local web server, spawning `ollama` / `llama.cpp` for export.
- **Policy risk:** the Store policy forbids changing or extending described functionality by downloading code at runtime. Today Reflex runs `bun install` from npm on first start, so **the package must ship the built frontend and run offline from first launch.** Downloading model weights from Hugging Face is data, not code.
- **Pre-check:** run the Windows App Certification Kit before upload.
- **Sources:**
  - https://learn.microsoft.com/en-us/windows/apps/publish/publish-your-app/msix/app-package-requirements
  - https://learn.microsoft.com/en-us/answers/questions/505097/providing-restricted-capabilities-explanation-via

## Package identity (from Partner Center, product 9MVXLZVL3TMT)

```
Package/Identity/Name                 mcp-tool-shop.backpropagate
Package/Identity/Publisher            CN=5305D976-6952-4F00-9C21-3A5DB090359F
Package/Properties/PublisherDisplayName  mcp-tool-shop
Package Family Name                   mcp-tool-shop.backpropagate_yn6b8xqrexa5j
Store ID                              9MVXLZVL3TMT
```

The values are case-sensitive. Re-read the Publisher value on the Partner Center
"Product identity" page before packing.

## Rig tooling (checked 2026-10-01)

- **Developer Mode:** on.
- **Windows SDK 10.0.26100:** `C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\makeappx.exe` (signtool is in the same folder).
- **WACK:** `C:\Program Files (x86)\Windows Kits\10\App Certification Kit\appcert.exe`.
- **Hardware:** RTX 5090, Windows 11.
- **Watchdog:** the VRAM watchdog may be dead. Tell Mike before any GPU training test. Restart it with `pwsh -NoProfile -File E:\AI\training\_watchdog_start.ps1`.

## What has to be built

1. **Reflex runs from a writable working folder (product change, 1.8.2).**
   - `backprop ui` runs Reflex with cwd set to the installed package directory.
   - Reflex writes `.web/`, `.states/`, `reflex.lock/`, `uploaded_files/`, `.gitignore` and `requirements.txt` there. (Found 2026-10-01 while making Docker work; see PR #268, which works around it with per-path ownership in the image.)
   - The MSIX install folder (`C:\Program Files\WindowsApps\...`) is read-only.
   - Move Reflex's app root to a per-user folder: `%LOCALAPPDATA%\backpropagate\ui` on Windows, the XDG cache dir on Linux.
   - One idea to spike first: a staging dir that holds only an `rxconfig.py` with `app_module_import="backpropagate.ui_app.app"`. The prod path never calls `get_reload_paths`.
   - **Watch for MSIX file-system virtualization:** AppData writes from a packaged app land in the package's private store. That is fine, but verify it.
   - The fix also simplifies the Docker image: drop the per-path chown once it lands.
2. **Frontend prebuilt at package time.**
   - Ship the compiled frontend and `node_modules` in the package.
   - First launch copies them into the working folder and runs with no network: no `bun install` from the registry.
   - Keep bun pinned and hash-checked, as `docker/fetch_bun.py` does. Use the Windows build for this.
   - Measure first-launch time.
3. **Self-contained layout.**
   - Embedded CPython 3.12 (python.org embeddable zip, hash-pinned) with `site-packages` from `uv.lock` (`--extra ui`, CUDA torch).
   - `uv export` with hashes, as the Dockerfile does.
   - No absolute paths baked into launchers: the WindowsApps path changes with every version.
4. **Entry points.**
   - **Start-menu tile:** console + browser, per the decision above. `uap10:Parameters` lets an Application point at `python.exe` with arguments, but a tiny launcher may be cleaner. Your call.
   - `backprop ui` today prints the token URL but does not open a browser, so you will need a flag such as `--open-browser`.
   - **Terminal command:** an App Execution Alias (`uap5:AppExecutionAlias`) for `backprop` and `backpropagate`.
5. **Build script.**
   - `scripts/build_msix.py`: assemble the layout, write `AppxManifest.xml` from the identity above and the version from pyproject, generate the tile and Store logos from the repo logo, then `makeappx pack`.
   - Version mapping: `1.8.2` becomes `1.8.2.0`.
   - Optional later: a release-only Windows workflow (release-triggered workflows don't count toward the 2-workflow cap).
6. **Local verification.**
   - Self-sign and sideload, or `Add-AppxPackage -Register` an unpacked layout in Developer Mode.
   - Then check: the tile starts the UI and the token works in the browser; `backprop info` and `backprop --help` from a fresh terminal; a short real training run (tell Mike first, per the watchdog note); GGUF export; uninstall leaves nothing behind outside AppData.
   - Run WACK and fix what it reports.
7. **Listing (lead plus Director).**
   - The lead fills Partner Center with Mike's permission: description, at least one screenshot (1366×768 or larger), category Developer tools, free, age-rating questionnaire, privacy URL, and the `runFullTrust` justification.
   - **Mike presses Submit.** Submission is irreversible: it goes to certification.

## Open questions for you

- **Unsloth.** Can't be an install-time choice. Options:
  - (a) Ship without it. Recommended for v1: the Triton builds it needs on Windows break often.
  - (b) An in-app "Enable Unsloth" that pip-installs into a per-user venv. Check this against the Store's dynamic-code policy before building it, and quote the policy section.
  - (c) An MSIX optional package.

  Recommend one with reasons. Mike decides.
- **Console vs no console for the tile.** Is a console window acceptable to certification and users? (The Director chose it; confirm nothing in certification objects.)
- **CUDA torch size.** Measure the package. The 25 GB cap is not a concern, but download size is.
- **Where `ollama` and `llama.cpp` come from.** Export paths call external tools. Check which features work inside the package, and say so in the listing.

## Repo rules you must follow

- **PRs as the bot.** Open PRs as the `mcp-tool-shop-bot` GitHub App. Token helper: `C:\Users\mikey\.config\mcp-tool-shop-bot\mint_token.py`; recipe in the lead's memory `backpropagate-bot-review-flow.md`. Never print the token or the key.
  - Never approve a PR with Mike's login. Mike approves on GitHub.
  - `main` needs 1 approval and 16 strict checks. Merge one PR at a time, up to date with main.
- **Public docs are lead-written.** README, CHANGELOG, handbook and SECURITY belong to the lead. List the doc changes you need instead of making them.
- **README translations.** README changes need local TranslateGemma translations before a release.
- **Atlas.** After code changes run `npx --yes @dogfood-lab/atlas@1.24.0 check`. If it's red after an intended change, run `map` and commit `atlas/`. Never hand-edit `atlas/`.
- **Drift check.** `scripts/check_doc_drift.py` must pass: new flags and env vars need handbook rows (the lead writes them; tell the lead).
- **Working copies are CRLF.** The venv `E:/AI/backpropagate/.venv` is an editable install of the main checkout; set `PYTHONPATH=<worktree>` when running from a worktree.
- **Known rig-only test failure:** `TestStageCHfTokenCap`.
- **Subagents** run on Sonnet (`model: "sonnet"`).
- **No Ollama Cloud.** Verification is proportional: tests and WACK, no verifier panels.

## Irreversible steps and their undo

| Action | Undo | Owner |
|---|---|---|
| Upload a package to a Partner Center submission | Delete the package from the draft submission | lead |
| Submit for certification | Cancel the submission before it publishes. After publishing: make the product unavailable, or submit a higher version | Mike |
| Publish 1.8.2 to PyPI/npm | Yank on PyPI / deprecate on npm, then ship 1.8.3 | lead |
