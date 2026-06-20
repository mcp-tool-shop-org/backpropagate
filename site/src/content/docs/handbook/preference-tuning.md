---
title: Preference tuning (ORPO / SimPO / KTO)
description: When to use SFT vs ORPO vs SimPO vs KTO — paired vs unpaired data, the VRAM envelope, data shapes, key hyperparameters, and the cited papers.
sidebar:
  order: 2.7
---

Backpropagate's training objective is selected with one knob — `method` (kwarg `Trainer(method=...)`, CLI `--method`, env `BACKPROPAGATE_TRAINING__METHOD`). It is one of four values:

| `method` | Stage | Data shape | Reference model | VRAM envelope |
|----------|-------|------------|-----------------|---------------|
| `sft` *(default)* | supervised | any chat format (ShareGPT / Alpaca / OpenAI / ChatML) | — | baseline |
| `orpo` | preference, reference-free | **paired** `{prompt, chosen, rejected}` | none (monolithic) | ≈ SFT |
| `simpo` | preference, reference-free | **paired** `{prompt, chosen, rejected}` | none (length-normalized reward) | **tightest of the paired methods** |
| `kto` | preference, binary feedback | **unpaired** `{prompt, completion, label}` | the frozen base (no second model) | ≈ SFT (16 GB) |

All three preference methods are **reference-free in practice** on a 16 GB card: ORPO and SimPO need no reference model at all, and KTO uses the frozen LoRA base as its own reference (TRL's `KTOTrainer` disables the adapter to compute reference logprobs), so no second model copy is loaded. That is the whole reason these four — and not classic DPO/PPO — are what Backpropagate ships: they fit the single-consumer-GPU envelope.

ORPO shipped in v1.5; **SimPO and KTO shipped in v1.6.**

## Which one should I use?

- **Start with `sft`.** If you have instruction → response data and no explicit "this answer is better than that one" signal, supervised fine-tuning is the right tool. Preference methods do not replace SFT — they refine a model on *comparative* signal.
- **Use `orpo` or `simpo` when you have paired preferences** — two candidate responses to the same prompt, one marked `chosen` and one `rejected`. Both are single-stage and reference-free.
  - Reach for **`simpo`** when VRAM is tightest: SimPO's length-normalized reward removes the per-token length bias without any reference model, and it is the leanest paired objective Backpropagate offers.
  - Reach for **`orpo`** when you want the odds-ratio formulation that folds the SFT loss and the preference penalty into one term (it tends to be a gentle, stable default for paired data).
- **Use `kto` when your feedback is binary and unpaired** — you have a pile of `(prompt, completion)` rows each tagged simply "good" (`label: true`) or "bad" (`label: false`), with **no requirement that a good and bad response share a prompt**. This is the realistic shape for thumbs-up/thumbs-down product telemetry, where you almost never have two responses to the *same* prompt. KTO is the unpaired / binary-feedback method.

## Data shapes

The format auto-detector keys off the columns present, so you usually do not declare the shape — you just point Backpropagate at the file and pick the `method`.

### Paired preference data (ORPO, SimPO)

One object per line, each carrying both a `chosen` and a `rejected` completion. `prompt` is optional (it can live inside the `chosen`/`rejected` message lists for the implicit-prompt case). Each value may be a plain string **or** an OpenAI-style message list:

```json
{"prompt": "Explain backpropagation to a 10-year-old.",
 "chosen": "Imagine you guessed an answer and a friend tells you how far off you were, so next time you guess a little closer. Backprop is the computer doing that, over and over.",
 "rejected": "Backpropagation is the reverse-mode automatic differentiation of the loss with respect to the parameters."}
```

Only rows that carry **both** `chosen` and `rejected` are kept for preference training (`DatasetLoader.to_preference_dataset`). A row missing one side is dropped, so a mixed file contributes only its real preference rows.

> Note: a `{prompt, chosen, rejected}` row trained under `method='sft'` is still valid — SFT renders `prompt → chosen` and deliberately drops `rejected`. The `rejected` column only carries signal under a preference method.

### Unpaired binary-feedback data (KTO)

One object per line: a `prompt`, a single `completion`, and a boolean `label` (`true` = desirable, `false` = undesirable). The ints `0` / `1` are accepted and coerced to `bool`; a float like `1.0` is **not** a valid label (it is read as a numeric score, not a binary flag) and a string class label is not a KTO label either.

```json
{"prompt": "Write a commit message for a one-line typo fix.", "completion": "fix typo in README", "label": true}
{"prompt": "Write a commit message for a one-line typo fix.", "completion": "Various changes and improvements across the codebase.", "label": false}
{"prompt": "Summarize the meeting in one sentence.", "completion": "We agreed to ship Friday and Jia owns the rollback plan.", "label": true}
```

The KTO converter (`DatasetLoader.to_kto_dataset`) emits exactly the columns `{prompt, completion, label}`. Rows without a `completion` **and** a boolean (or 0/1) `label` are dropped. There is no pairing requirement — desirable and undesirable rows are independent.

## Key hyperparameters and defaults

Every knob below is inert unless its method is selected. Each maps to a CLI flag (`--simpo-beta`, `--kto-beta`, …), a `Trainer(...)` kwarg of the same name, and an env var (`BACKPROPAGATE_TRAINING__SIMPO_BETA`, …) — see [Environment variables → Training](/backpropagate/handbook/env-vars/#training) for the full env table.

### ORPO

| Knob | Default | Meaning |
|------|---------|---------|
| `orpo_beta` | `0.1` | Odds-ratio weight (the ORPO "lambda" / TRL `ORPOConfig.beta`). Must be > 0 — a non-positive value silently degenerates ORPO back to plain SFT (zero) or trains toward the *rejected* completion (negative), so both are rejected at construction with `CONFIG_INVALID_SETTING`. `0.1` is the paper's headline setting. |

### SimPO

SimPO has no dedicated TRL trainer — it is TRL's `CPOTrainer` / `CPOConfig` driven with `loss_type="simpo"` and **`cpo_alpha=0.0` forced** (a non-zero `cpo_alpha` would be "CPO-SimPO", a different method; Backpropagate always means *pure* SimPO). It consumes the same paired `{prompt, chosen, rejected}` data as ORPO.

| Knob | Default | Meaning |
|------|---------|---------|
| `simpo_beta` | `2.0` | Reward-scaling temperature (`CPOConfig.beta`). The cross-setup safe floor from the paper. Any finite value is admissible. |
| `simpo_gamma` | `1.0` | Target reward margin (`gamma`; absolute, = `beta`×0.5 at the default beta). Must be > 0 (`CONFIG_INVALID_SETTING`). |

A `gamma/beta` ratio **above 1.0** over-weights the margin relative to the reward scale and risks repetitive / degenerate output — that is a soft signal, so it only emits a WARN (the run is still launchable). The paper pins `gamma` at roughly `beta`×0.5; keep the ratio ≤ 1.0.

### KTO

KTO is **LoRA-mode-only** in v1.6 (`kto` + `mode='full'` is rejected at construction — the frozen base must exist to serve as the reference). TRL's `KTOTrainer` / `KTOConfig`.

| Knob | Default | Meaning |
|------|---------|---------|
| `kto_beta` | `0.1` | Prospect-theory loss temperature (`KTOConfig.beta`). |
| `kto_desirable_weight` | `1.0` | Loss weight on desirable (`label=true`) examples. Must be > 0. |
| `kto_undesirable_weight` | `1.0` | Loss weight on undesirable (`label=false`) examples. Must be > 0. |

**The desirable/undesirable weights you set are a starting point, not the final ratio.** The trainer auto-rebalances the *effective* weights from your dataset's actual label counts so the desirable:undesirable contribution lands in the `[1:1, 4:3]` band recommended by the KTO authors. This corrects for class imbalance (a dataset that is 90% "good" rows would otherwise drown the "bad" signal). A zero or negative weight is still rejected at construction (`CONFIG_INVALID_SETTING`) — start from positive values and let the trainer balance.

## Learning rate (auto-lowered)

Preference objectives are far more LR-sensitive than SFT. When you do **not** pass an explicit learning rate, Backpropagate auto-selects a method-appropriate default:

- **SFT** — a dataset-size ladder (small `5e-4` / medium `2e-4` / large `1e-4`).
- **ORPO** — a lower dataset-size ladder (`2e-5` / `1e-5` / `5e-6`); the odds-ratio penalty is unstable at SFT magnitudes ([Hong, Lee & Thorne 2024, arXiv:2403.07691](https://arxiv.org/abs/2403.07691)).
- **SimPO** and **KTO** — a **fixed `1e-6` anchor** at every dataset size. SimPO degrades to repetitive output at LR ≥ `1e-5` ([Meng et al. 2024](https://arxiv.org/abs/2405.14734)) and KTO's published runs sit at `1e-6` ([Ethayarajh et al. 2024](https://arxiv.org/abs/2402.01306)). These anchors are published settings, **not** scaled off the SFT base LR.

If you pass an explicit `--lr` / `learning_rate=...`, it wins — but for SimPO a value ≥ `1e-5` is clamped down with a warning, because high LR is the documented SimPO failure mode.

## Verifying the result

After a preference run, score the adapter against a held-out reference set rather than eyeballing loss — loss is a weak proxy. The eval harness computes deterministic, judge-free task metrics and can gate a merge on non-regression:

```bash
# Carve a held-out reference split, then score the run on exact-match + token-F1
backprop data split prefs.jsonl --heldout-ratio 0.1 --seed 0
backprop eval <run_id> --references prefs.heldout.jsonl \
  --metric normalized_exact_match --metric token_f1
```

See [Recipes](/backpropagate/handbook/recipes/) for the full SimPO / KTO / eval-gate recipes.

## Papers

- **ORPO** — Hong, Lee & Thorne, *ORPO: Monolithic Preference Optimization without Reference Model* (2024). [arXiv:2403.07691](https://arxiv.org/abs/2403.07691)
- **SimPO** — Meng, Xia & Chen, *SimPO: Simple Preference Optimization with a Reference-Free Reward* (2024). [arXiv:2405.14734](https://arxiv.org/abs/2405.14734)
- **KTO** — Ethayarajh, Xu, Muennighoff, Jurafsky & Kiela, *KTO: Model Alignment as Prospect Theoretic Optimization* (2024). [arXiv:2402.01306](https://arxiv.org/abs/2402.01306)

## See also

- [Training](/backpropagate/handbook/training/) — the base `Trainer` surface, dataset formats, callbacks.
- [Recipes](/backpropagate/handbook/recipes/) — paste-and-run SimPO / KTO / eval snippets.
- [Environment variables → Training](/backpropagate/handbook/env-vars/#training) — the `BACKPROPAGATE_TRAINING__*` knobs.
- [CLI reference](/backpropagate/handbook/cli-reference/) — every `--method` / `--simpo-*` / `--kto-*` flag.
