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
// Tracked: docs/V1_3_BRIEF.md (D2 SPLIT — Wave 1 hotfix, Wave 5/6 deletes
// .github/workflows/release-binaries.yml + the .spec files).
// ---------------------------------------------------------------------------

process.stderr.write(
  "npm distribution of backpropagate is deprecated.\n" +
  "\n" +
  "Install via one of the supported channels:\n" +
  "  pipx install backpropagate   (recommended — isolated venv)\n" +
  "  uv tool install backpropagate\n" +
  "  pip install backpropagate\n" +
  "\n" +
  "Optional extras (UI / GGUF export / monitoring) are documented at:\n" +
  "  https://github.com/mcp-tool-shop-org/backpropagate#installation\n"
);
process.exit(2);
