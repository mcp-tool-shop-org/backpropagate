---
title: Privacy
description: What backpropagate stores on your machine, what it sends over the network and when, and how to remove all of it.
sidebar:
  order: 9.5
---

backpropagate is a local tool. It has no account, no analytics and no telemetry, and it does not send your data anywhere unless you ask it to do something that needs the network, such as downloading a model or pushing one to the Hugging Face Hub. This page lists every case.

It applies to every way of installing backpropagate: `pip`, Docker, and the Microsoft Store package.

## What leaves your machine, and when

| When | Where it goes | What is sent |
|---|---|---|
| You train on, or download, a model or dataset by its Hugging Face name | Hugging Face Hub (`huggingface.co`) | Requests for that model or dataset, with your Hugging Face token if one is set. |
| You push a model (`backprop push`, `push_to_hub(...)`, or an export with a Hub repo) | Hugging Face Hub | The model files you chose to push, to the repository you named. |
| You install an experiment tracker and leave tracking on its default (`report_to="auto"`) | That tracker's service, e.g. Weights & Biases | Training metrics and run configuration. **Off unless a tracker is installed:** the `[monitoring]` extra installs Weights & Biases; the Microsoft Store package includes no tracker. Pass `report_to="none"` to turn it off. |
| You export to Ollama | The Ollama server **on your own machine** (`localhost:11434`) | The exported model. Nothing leaves the machine. |
| You start the UI with `backprop ui --share` | Cloudflare's tunnel service (`*.trycloudflare.com`) | Your UI's traffic, so that people with the URL and password can reach it. Only with `--share`; see [Security](/backpropagate/handbook/security/). |
| The UI starts | The Python Package Index (`pypi.org`) | The UI framework (Reflex) checks whether a newer Reflex release exists. Set `REFLEX_CHECK_LATEST_VERSION=false` to turn this off. The check gives up quietly after two seconds when you are offline. |

Since 1.8.1 the UI framework's own usage telemetry is switched off (`telemetry_enabled=False`). Versions before 1.8.1 left it on, and each UI launch sent an anonymous usage event to Reflex's analytics service.

By default the web UI listens only on `127.0.0.1`, so other machines cannot reach it, and it requires the token printed at startup.

## Your Hugging Face token

backpropagate reads your token from, in order: the `--token` flag, `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, or the file `huggingface-cli login` writes (`~/.cache/huggingface/token`). It uses the token only for requests to the Hugging Face Hub. It does not copy it anywhere else. In normal (non-verbose) mode, error output is scrubbed of tokens and keys before it is printed.

## What is stored on your machine

| What | Where |
|---|---|
| Training outputs, checkpoints and `run_history.json` | The output folder you choose (`./output` by default). |
| Files the web UI saves (adapters, exports, converted datasets) | `~/.backpropagate/ui-outputs`, or `BACKPROPAGATE_UI__OUTPUT_DIR`. |
| The web UI's build and state files | `%LOCALAPPDATA%\backpropagate\ui\` on Windows, `~/.cache/backpropagate/ui/` on Linux and macOS (or under `$XDG_CACHE_HOME`), or `BACKPROPAGATE_UI_WORKDIR`. |
| The JavaScript runtime the UI uses (bun) | `%LOCALAPPDATA%\reflex\` on Windows, `~/.local/share/reflex/` on Linux, `~/Library/Application Support/reflex/` on macOS. |
| The UI's launch token, while the UI runs | `%LOCALAPPDATA%\backpropagate\session-<port>.lock` on Windows, `$XDG_RUNTIME_DIR/backpropagate/` on Linux, `~/Library/Application Support/backpropagate/` on macOS. Deleted when the UI stops. |
| Downloaded models and datasets | The Hugging Face cache, `~/.cache/huggingface` (or `HF_HOME`). Shared with other Hugging Face tools. |
| Log files | Only if you set `BACKPROPAGATE_LOG_FILE`. |

Your datasets and models stay where you put them. backpropagate does not upload them unless you push.

## Removing everything

Uninstalling backpropagate (`pip uninstall backpropagate`, or uninstalling the Microsoft Store app) removes the program but not the files listed above, like most desktop tools. To remove them too, delete:

- `%LOCALAPPDATA%\backpropagate` and `%LOCALAPPDATA%\reflex` on Windows; `~/.cache/backpropagate` and `~/.local/share/reflex` on Linux; `~/.cache/backpropagate`, `~/Library/Application Support/backpropagate` and `~/Library/Application Support/reflex` on macOS
- `~/.backpropagate`
- your output folders
- models in the Hugging Face cache that you no longer want (`huggingface-cli delete-cache`). Other Hugging Face tools share this cache.

## Questions

Open an issue at [github.com/mcp-tool-shop-org/backpropagate/issues](https://github.com/mcp-tool-shop-org/backpropagate/issues). To report a security problem privately, follow [SECURITY.md](https://github.com/mcp-tool-shop-org/backpropagate/blob/main/SECURITY.md).
