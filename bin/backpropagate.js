#!/usr/bin/env node
"use strict";

const { spawnSync } = require("child_process");
const path = require("path");
const fs = require("fs");
const os = require("os");

// ---------------------------------------------------------------------------
// Linux: bootstrap a private venv and run backpropagate from PyPI.
//
// PyTorch's libtorch_cpu.so is ~1.5 GB on x86_64 Linux. A PyInstaller
// --onefile binary lands at 4.16 GB — 2x GitHub's release asset limit.
// macOS/Windows don't hit this because Accelerate is system-provided and
// Windows torch is smaller.
//
// Instead of shipping a broken giant binary, we bootstrap a managed venv
// on first run and exec the pip-installed CLI from there.
// ---------------------------------------------------------------------------

const TOOL = "backpropagate";
const VERSION = "1.1.1";

if (process.platform === "linux") {
  // XDG-compliant install root
  const dataHome = process.env.XDG_DATA_HOME
    || path.join(os.homedir(), ".local", "share");
  const installRoot = process.env.BACKPROPAGATE_BOOTSTRAP_ROOT
    || path.join(dataHome, TOOL);
  const venvDir = path.join(installRoot, "venv");
  const metaPath = path.join(installRoot, "install.json");
  const venvBin = path.join(venvDir, "bin", TOOL);
  const venvPython = path.join(venvDir, "bin", "python3");

  // -- helpers --------------------------------------------------------------

  function findPython() {
    // Prefer python3, fall back to python (if it's 3.x)
    for (const cmd of ["python3", "python"]) {
      const r = spawnSync(cmd, ["--version"], {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "pipe"],
      });
      if (r.status === 0) {
        const ver = (r.stdout || r.stderr || "").trim();
        const match = ver.match(/Python\s+(\d+)\.(\d+)/);
        if (match && parseInt(match[1], 10) >= 3 && parseInt(match[2], 10) >= 10) {
          return cmd;
        }
      }
    }
    return null;
  }

  function readMeta() {
    try {
      return JSON.parse(fs.readFileSync(metaPath, "utf8"));
    } catch {
      return null;
    }
  }

  function writeMeta(obj) {
    fs.mkdirSync(installRoot, { recursive: true });
    fs.writeFileSync(metaPath, JSON.stringify(obj, null, 2) + "\n");
  }

  function fail(message, hint) {
    process.stderr.write(`\n${TOOL}: ${message}\n`);
    if (hint) process.stderr.write(`\n${hint}\n`);
    process.stderr.write("\n");
    process.exit(1);
  }

  // -- bootstrap ------------------------------------------------------------

  function bootstrap() {
    const forceReinstall = process.env.BACKPROPAGATE_FORCE_REINSTALL === "1";
    const meta = readMeta();

    const versionMatch = meta && meta.version === VERSION;
    const binaryPresent = fs.existsSync(venvBin);

    if (versionMatch && binaryPresent && !forceReinstall) {
      return; // fast path — venv exists, version matches, no forced reinstall
    }

    // Decide what to tell the user
    if (forceReinstall) {
      process.stderr.write(`Forced reinstall requested (BACKPROPAGATE_FORCE_REINSTALL=1).\n`);
    } else if (meta && !versionMatch) {
      process.stderr.write(
        `Updating ${TOOL}: ${meta.version} -> ${VERSION}\n`
      );
    } else if (meta && !binaryPresent) {
      process.stderr.write(
        `Repairing ${TOOL}: binary missing, reinstalling ${VERSION}...\n`
      );
    }

    // Find system Python
    const python = findPython();
    if (!python) {
      fail(
        "Python 3.10+ is required but not found.",
        "Install Python and try again:\n" +
        "  Ubuntu/Debian:  sudo apt install python3 python3-venv\n" +
        "  Fedora/RHEL:    sudo dnf install python3\n" +
        "  Arch:           sudo pacman -S python\n" +
        "  Or use pyenv:   https://github.com/pyenv/pyenv"
      );
    }

    // Recreate venv on force-reinstall (nuke corrupted state)
    if (forceReinstall && fs.existsSync(venvDir)) {
      process.stderr.write("Removing existing venv...\n");
      fs.rmSync(venvDir, { recursive: true, force: true });
    }

    // Create venv if missing
    if (!fs.existsSync(venvPython)) {
      if (!meta) {
        process.stderr.write(
          `First run on Linux: setting up local Python environment for ${TOOL}...\n`
        );
      }
      fs.mkdirSync(installRoot, { recursive: true });
      const venvResult = spawnSync(python, ["-m", "venv", venvDir], {
        stdio: "inherit",
      });
      if (venvResult.status !== 0) {
        fail(
          "Failed to create Python virtual environment.",
          "The venv module may be missing. Try:\n" +
          "  Ubuntu/Debian:  sudo apt install python3-venv\n" +
          "  Fedora/RHEL:    sudo dnf install python3-libs"
        );
      }
    }

    // Install or upgrade backpropagate
    // --force-reinstall needed for repair (binary missing but pip metadata intact)
    // and force-reinstall (nuked venv rebuilt). Without it, pip sees the same
    // version in metadata and skips recreating the entrypoint script.
    const needsForce = !binaryPresent || forceReinstall;
    const pipArgs = ["-m", "pip", "install", "--quiet", "--upgrade"];
    if (needsForce) pipArgs.push("--force-reinstall");
    pipArgs.push(`${TOOL}==${VERSION}`);

    process.stderr.write(`Installing ${TOOL} ${VERSION}${needsForce ? " (force)" : ""}...\n`);
    const pipResult = spawnSync(venvPython, pipArgs, { stdio: "inherit" });
    if (pipResult.status !== 0) {
      fail(
        `pip install failed (exit ${pipResult.status}).`,
        "Check your network connection and try again.\n" +
        "You can also install manually:\n" +
        `  pip install ${TOOL}==${VERSION}`
      );
    }

    // Verify the binary actually exists after install
    if (!fs.existsSync(venvBin)) {
      fail(
        `Installation completed but ${TOOL} binary not found at expected path.`,
        `Expected: ${venvBin}\n` +
        "Try installing manually:\n" +
        `  pipx install ${TOOL}`
      );
    }

    // Record metadata
    writeMeta({
      version: VERSION,
      installedAt: new Date().toISOString(),
      python,
      venvDir,
    });

    process.stderr.write("Ready.\n");
  }

  // -- exec -----------------------------------------------------------------

  bootstrap();
  const result = spawnSync(venvBin, process.argv.slice(2), { stdio: "inherit" });
  if (result.error) {
    fail(`Failed to execute ${TOOL}: ${result.error.message}`);
  }
  process.exit(result.status ?? 1);
}

// ---------------------------------------------------------------------------
// macOS + Windows: download and launch prebuilt binary via npm-launcher
// ---------------------------------------------------------------------------

process.env.MCPTOOLSHOP_LAUNCH_CONFIG = JSON.stringify({
  toolName: TOOL,
  owner: "mcp-tool-shop-org",
  repo: "backpropagate",
  version: VERSION,
  tag: `v${VERSION}`,
});

require("@mcptoolshop/npm-launcher/bin/mcptoolshop-launch.js");
