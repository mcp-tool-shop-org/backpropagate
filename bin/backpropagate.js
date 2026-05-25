#!/usr/bin/env node
"use strict";

// ---------------------------------------------------------------------------
// npm distribution of backpropagate is deprecated as of v1.3.
//
// History: v1.0–v1.2 shipped PyInstaller binaries from a GitHub Release,
// pulled via @mcptoolshop/npm-launcher. Linux bootstrapped a managed venv
// instead because libtorch_cpu.so blew past GitHub's 2GB release-asset cap.
// The binary build pipeline failed three consecutive times in v1.2.0 and the
// release tag has zero attached assets — the launcher would 404 on download.
//
// Rather than ship a broken installer, the npm shim now prints install
// guidance for the supported channels. The package stays published so this
// message reaches operators who still have `npm install -g backpropagate`
// in their tooling.
//
// Tracked: D2 SPLIT in the v1.3 brief. Wave 1 landed the friendly-error
// hotfix; Wave 3.5 deleted .github/workflows/release-binaries.yml; Wave 6a
// removed the PyInstaller .spec files at the repo root and added the v1.2.x
// → v1.3 handbook migration page that walks operators from the
// pre-deprecation `npm install -g backpropagate` install line to the
// supported channels. The migration is complete in v1.3.
// ---------------------------------------------------------------------------

// BRIDGE-B (Stage C humanization): the shim's WHOLE job is the friendly-error
// path. Three rules the message has to satisfy:
//   1. Name the next step (every error names the next step).
//   2. Make the install commands copy-paste-runnable on the operator's host
//      (no shell quoting that fails on cmd.exe, no `sudo` prefix that
//      misleads on Windows).
//   3. Stay calibrated — the v1.2 PyInstaller-bin build failed 3 times so
//      v1.3 redirected operators to PyPI / pipx / uv; the message names the
//      channels in preference order (isolated > shared > root-installable).
process.stderr.write(
  "npm distribution of backpropagate is deprecated as of v1.3.\n" +
  "\n" +
  "Next step — install from PyPI via one of these channels (pick one):\n" +
  "  pipx install backpropagate         # recommended on macOS/Linux (isolated venv, on PATH)\n" +
  "  uv tool install backpropagate      # recommended on Windows (uv handles venv + PATH)\n" +
  "  pip install backpropagate          # plain pip (use inside a venv)\n" +
  "\n" +
  "Verify the install:  backprop --version\n" +
  "\n" +
  "Optional extras (Reflex UI, GGUF export, monitoring) — pick one bundle:\n" +
  "  pipx install 'backpropagate[standard]'    # unsloth + ui (recommended)\n" +
  "  pipx install 'backpropagate[full]'        # everything\n" +
  "  pipx install 'backpropagate[ui]'          # just the Reflex web UI\n" +
  "\n" +
  "Getting started + extras documentation:\n" +
  "  https://mcp-tool-shop-org.github.io/backpropagate/handbook/getting-started/\n" +
  "Source + issues:\n" +
  "  https://github.com/mcp-tool-shop-org/backpropagate\n"
);
process.exit(2);
