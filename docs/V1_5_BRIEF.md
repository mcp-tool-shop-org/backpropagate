# Backpropagate v1.5 — Trajectory Brief

**Status:** DRAFT for director review · **Date:** 2026-05-30 · **Author:** Advisor (dogfood-swarm study-swarm)
**Current release:** v1.4.0 (PyPI, Production/Stable) · **Target:** v1.5.0

This brief is the dispatch artifact for the v1.5 feature pass. It is grounded in a five-agent study swarm (research-grounded-advisor-protocol) run 2026-05-30. Every load-bearing decision below traces to a cited finding in the [Research grounding](#research-grounding) section. The README's "What you can fine-tune on a 16GB consumer GPU" table already pre-commits to "see V1_5_BRIEF when posted" — this is that document.

---

## 1. Strategic frame

Backpropagate is **the fine-tuning backbone for model customization across the studio's game pipeline** — it is not just a public tool, it is internal infrastructure for producing custom small models. That dual role raises the value of any feature that improves *our own* model-customization loop (dataset quality, eval, reasoning distillation), not only features that widen the public addressable market.

The study swarm's meta-finding: **the product identity is correct and intact** — 3-line API, single consumer GPU, Windows-first, ship-to-Ollama. The field did not invalidate the identity. It did two things:

1. **Made two stated "NOT FOR" boundaries obsolete** — reference-free preference tuning, and Apple-Silicon training.
2. **Broke one committed roadmap item** — the AQLM-2bit → Mixtral-8x7B@16GB v1.5 plan is technically infeasible *and* would break the signature GGUF export pipeline.

v1.5 is therefore a **correct-the-course + close-the-gaps + extend-the-moat** release, in the db-cluster mold (already-shipped → harden → feature → composed re-audit → ship).

---

## 2. Research grounding {#research-grounding}

Each finding: one-sentence claim · source · URL · design implication. ⚠️ marks an assumption the evidence challenges.

### A. The 16 GB memory / quantization frontier

1. **AQLM adapters cannot be merged into the 2-bit base weights.** HuggingFace PEFT quantization docs + Egiazarian et al. 2024 (arXiv:2401.06118). https://huggingface.co/docs/peft/developer_guides/quantization — the doc states merging LoRA into AQLM weights "is not possible." **Implication:** an AQLM path could only ship base+adapter, never a merged GGUF — it breaks the "ship it" export pipeline at its core.
2. **⚠️ Mixtral-8x7B does not fit 16 GB even for inference.** Eliseev & Mazur 2023 (arXiv:2312.17238, "Fast Inference of MoE with Offloading"). https://arxiv.org/abs/2312.17238 — INT4 Mixtral needs ~24–27 GB of weights; 16 GB requires expert-offloading, an inference-only trick. No 2024–26 source fine-tunes it on 16 GB. **Implication: kill the AQLM-Mixtral v1.5 plan.**
3. **The credible 16 GB stretch target is a modern sparse MoE via ordinary 4-bit QLoRA.** Unsloth 2025 docs — Qwen3-30B-A3B (30.5 B total / 3.3 B active) fine-tunes in ~17.5 GB. https://docs.unsloth.ai/ **Implication:** re-aim the stretch story at "biggest modern MoE on 16 GB" via standard NF4 QLoRA — 30 B-class total params, and the merge→GGUF path still works.
4. **FP8 training is the real RTX-50-series headroom lever and PEFT supports LoRA-on-FP8.** torchao 2025 (arXiv:2507.16099) + HF PEFT Transformer-Engine LoRA. https://arxiv.org/abs/2507.16099 — Blackwell 5th-gen tensor cores; ~1.4× throughput, up to 60% less model memory; mergeable. **Implication:** an FP8 compute path exploits the tested-on-5080/5090 moat with exportable results — higher payoff than 2-bit.
5. **Paged 8-bit AdamW + gradient checkpointing remain the highest-leverage, lowest-risk VRAM wins.** Dettmers et al. 2023 (QLoRA, arXiv:2305.14314). https://arxiv.org/abs/2305.14314 **Implication:** ensure these are the *defaults* (the recent CPU-runner commits already touch `adamw_8bit`); this is the floor that makes the 30B-A3B@16GB target reliable.
6. **HQQ / torchao are more PEFT-mature and merge-friendly than AQLM for sub-4-bit/8-bit LoRA.** HF PEFT quantization docs 2026. **Implication:** if a below-4-bit path is ever wanted, HQQ (first-class `HqqConfig`) or torchao-int8 (mergeable) beats AQLM — but neither is needed for v1.5 given finding 3.
7. **GaLore is a pre-training tool, not a fine-tuning win.** Zhao et al. 2024 (arXiv:2403.03507). https://arxiv.org/abs/2403.03507 — online SVD costs up to 80% of training time; "minimal improvements over LoRA." Q-GaLore (arXiv:2407.08296) pretrains 7B from scratch on 16 GB. **Implication:** do NOT adopt GaLore for fine-tuning; Q-GaLore is a *future* "pretrain-from-scratch on 16 GB" mode, distinct from the LoRA core.

### B. The preference-tuning boundary

8. **ORPO folds preference alignment into SFT as a single stage with no reference model.** Hong, Lee & Thorne 2024 (arXiv:2403.07691, "ORPO: Monolithic Preference Optimization without Reference Model"). https://arxiv.org/abs/2403.07691 — replaces the NLL loss with NLL + odds-ratio over (chosen, rejected). **Implication:** ORPO fits the existing 3-line single-stage API verbatim — a loss-term swap, no new training loop, no reference model.
9. **ORPO substantially beats SFT-only on the benchmarks solo finetuners care about.** Hong, Lee & Thorne 2024 — Mistral-ORPO-β: 12.20% AlpacaEval 2.0, 66.19% IFEval, 7.32 MT-Bench; won up to ~85% head-to-head vs SFT. **Implication:** crossing the line measurably raises output quality, not just feature count.
10. **SimPO is reference-free and cheaper than DPO (~20% less runtime, ~10% less GPU memory) while beating it.** Meng, Xia & Chen 2024 (arXiv:2405.14734, NeurIPS 2024). https://arxiv.org/abs/2405.14734 **Implication:** a good "advanced" second tier after ORPO.
11. **KTO aligns from binary thumbs-up/down (unpaired) data, matching/exceeding DPO 1B–30B.** Ethayarajh et al. 2024 (arXiv:2402.01306, ICML 2024). https://arxiv.org/abs/2402.01306 **Implication:** fits the "I have logs of good/bad completions, not curated pairs" persona — pairs with the dataset auto-detect philosophy.
12. **The DPO reference-model VRAM burden is exactly what reference-free methods eliminate.** Wang et al. 2024 survey (arXiv:2410.15595). https://arxiv.org/abs/2410.15595 **Implication:** the "DPO is expensive" rationale behind the current boundary does not apply to ORPO/SimPO — on 16 GB the envelope is essentially the SFT envelope.
13. **⚠️ All four single-GPU competitors ship preference tuning by default.** Unsloth (DPO/ORPO/KTO/SimPO notebooks), Axolotl (DPO/IPO/KTO/ORPO/SimPO), torchtune (`lora_dpo_single_device`), LLaMA-Factory (ORPO/KTO/SimPO since 2024). **Implication:** a 2026 solo finetuner *is* surprised when a tool can't do ORPO/DPO — "no preference tuning" is obsolete as written.
14. **Online RL (PPO/GRPO/RLVR) remains genuinely heavyweight.** "Post-Training in 2026: GRPO, DAPO, RLVR & Beyond" (llm-stats.com 2026). **Implication:** keep the boundary on online RL — that scope discipline is real and on-brand.

### C. The LoRA-variant default

15. **DoRA's quality advantage over LoRA collapses to ~0.3% at rank-256 — exactly where the default sits.** Liu et al. 2024 (arXiv:2402.09353): r=4 gap large, r=128/256 gap only +0.22%/+0.33%. https://arxiv.org/abs/2402.09353 **Implication:** keep DoRA opt-in (already exposed); its value is the low-rank `fast` preset, not the rank-256 default.
16. **DoRA costs ~5–10% extra VRAM and ~10–12% throughput.** Spheron 2026 PEFT guide. **Implication:** a default DoRA flip would silently shrink the 16 GB envelope for a sub-0.5% gain — disqualified as a default.
17. **The cited authority validates the existing default and did not test variants.** Thinking Machines 2025 ("LoRA Without Regret"): rank-256 LoRA ≈ FullFT; attention-only underperforms; PiSSA left as an open question. https://thinkingmachines.ai/blog/lora/ **Implication:** keep vanilla LoRA, all-linear, rank-256.
18. **⚠️ The highest-leverage lever is the learning rate, not the adapter type.** "Learning Rate Matters: Vanilla LoRA May Suffice" 2026 (arXiv:2602.04998) — much of the reported DoRA/PiSSA/LoRA+ gain disappears once LoRA's LR is properly set. **Implication:** audit and document the default LR (TM advises a 10–100× multiple over full-FT) before chasing adapter swaps.
19. **rsLoRA (α/√r scaling) is the one variant whose benefit grows with rank, at zero inference cost.** Kalajdzievski 2023 (arXiv:2312.03732). https://arxiv.org/abs/2312.03732 — standard α/r throttles gradients at high rank. **Implication:** at rank-256 the default scaling may under-train the adapter; `use_rslora=True` is the single default-flip worth validating empirically.
20. **PiSSA/EVA/OLoRA init are mature one-flag PEFT options but alter save/export semantics.** EVA: arXiv:2410.07170; PiSSA: arXiv:2404.02948; LoRA+: arXiv:2402.12354. **Implication:** expose as documented opt-ins, never silent defaults — PiSSA/DoRA mutate the residual/magnitude the GGUF→Ollama path depends on.

### D. Competitive gap + obsolete boundaries

21. **⚠️ MLX-LM ships production LoRA/QLoRA/DoRA + full-FT on Apple Silicon across 9 families overlapping ours.** ml-explore / Apple 2026 (mlx-lm LORA.md). https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md — Mistral/Llama/Phi/Mixtral/Qwen2/Gemma/OLMo/InternLM2; same GGUF→Ollama export. **Implication:** the "no macOS training" boundary is obsolete — the blocker was CUDA-coupling, not macOS. An MLX backend is dogfoodable on the M5 Max.
22. **Apple unified memory removes the VRAM wall (≈2–4× slower steps).** TDS / InsiderLLM guides 2025–26. **Implication:** the Mac story is complementary to the RTX moat — "your RTX *or* your Mac, one API" is positioning no competitor cleanly owns (CUDA tools are CUDA-only; MLX tools are Mac-only).
23. **Unsloth's "direct Windows" still needs a Triton fork + WSL for GRPO/vLLM.** Unsloth issue #2395. https://github.com/unslothai/unsloth/issues/2395 **Implication:** the Windows moat is contested but NOT obsolete — make "works on bare Windows, no WSL, no Triton fork" the loudest headline.
24. **Reasoning-trace SFT (R1 distillation) is now baseline, and the easy half is pure SFT.** DeepSeek-AI 2025 (arXiv:2501.12948) — R1 distilled six dense models via pure SFT on 800K reasoning traces, no RL. https://arxiv.org/abs/2501.12948 **Implication:** own the easy half — chat-template `<think>` handling + trace-length filtering in `datasets.py`; high-leverage, no RL scope creep.
25. **⭐ Operators' #1 unmet pain is dataset quality/prep + eval, not the training loop.** Data-selection survey (arXiv:2402.05123) + application-ready data-prep survey 2026 (arXiv:2601.17058) — injected noise dropped precision 89%→72%; ~10K curated points match far larger sets. https://arxiv.org/abs/2402.05123 **Implication:** the biggest underserved, on-brand gap. Extend `datasets.py` (already has detect/dedupe/filter/curriculum) toward a data-quality report + a lightweight eval harness. **This is the next moat — competitors are recipe/RL-focused and won't chase it.**
26. **Multimodal/vision LoRA went mainstream across competitors but carries high collation complexity.** Axolotl multimodal docs (Llama-Vision/Qwen2-VL/Pixtral). **Implication:** real demand, but it would dilute the 3-line promise on 16 GB — keep as a boundary, or at most a single deliberate Qwen2.5-VL image-only recipe, lower priority than 24/25.

### E. Export/serving + continual-learning differentiators

27. **Ollama loads GGUF LoRA adapters natively via the `ADAPTER` instruction — no full merge required.** Ollama 2025 docs (Modelfile/Import). https://docs.ollama.com/modelfile **Implication:** add an `--format ollama-adapter` export (GGUF adapter + `ADAPTER` Modelfile) for instant hot-swap between adapters on one base.
28. **But Ollama warns adapters-on-quantized-bases degrade quality and force an unquantized base into memory.** Ollama import docs 2025. **Implication:** merged-then-quantized GGUF stays the **default** for low-VRAM serving; adapter-native ships as an **option**, not a replacement — the moat is intact, just widened.
29. **vLLM is the production multi-LoRA / hot-swap leader (runtime adapter loading).** vLLM 2025 docs. https://docs.vllm.ai/en/stable/features/lora/ **Implication:** server-tier, not the solo lane — emit a vLLM-compatible adapter dir + a copy-paste snippet so users *can* graduate; own the export, not the server.
30. **Qiao & Mahdavi 2025 ("Merge before Forget") is benchmarked against the right baselines but is fresh and single-group.** arXiv:2512.23017 — vs SeqLoRA/O-LoRA/InfLoRA/TIES/DARE/task-arithmetic/model-soups. https://arxiv.org/abs/2512.23017 **Implication:** keep SLAO and the "first adopter" claim, but gate it ("implements / first adopter," not "best") — the method is not yet independently validated.
31. **2025 evidence favors selectable + eval-gated merging over any single algorithm.** K-Merge (arXiv:2510.13537) — linear merge is a strong baseline, a similarity threshold decides merge-vs-new-slot; DELLA (arXiv:2406.11617) beats TIES by 3.6, DARE by 1.2. **Implication:** extend SLAO into a *merge framework* (TIES/DARE/linear/Qiao-Mahdavi options) + a drift/similarity gate + eval-gated merges — reframes the differentiator from "one paper" to "the only solo-GPU continual-LoRA toolkit with safe, gated, multi-strategy merging." The v1.4 **drift-gate-5-class** already provides the scaffolding.

---

## 3. The trajectory

### Tier 0 — Corrections (do regardless; mostly within the Health Pass)

- **T0.1 Retire the AQLM-Mixtral v1.5 plan.** Update the README envelope table + anti-pitch. Re-aim the stretch target at **Qwen3-30B-A3B via 4-bit QLoRA** (findings 1–3). *Doc + preset work; small.*
- **T0.2 Keep the rank-256 all-linear vanilla-LoRA default** (findings 15–17). No change; document the rationale.
- **T0.3 Audit + document the default learning rate** (finding 18). Likely higher-leverage than any adapter swap. *Investigation + doc; small.*
- **T0.4 Hold the boundaries that still hold:** online RL (PPO/GRPO), full-FT of 7B+, multi-node (finding 14). Keep, with evidence.

### Tier 1 — Own the moat / close glaring gaps

- **T1.1 ⭐ Dataset-quality + post-train eval loop** (finding 25). The next moat; directly serves the studio's own model-customization. *Effort: L.*
- **T1.2 ORPO** — close the preference-tuning gap; reference-free, single-stage, fits the 3-line API as a loss swap (findings 8–13). *Effort: M.*
- **T1.3 Sharpen the "Windows without WSL/Triton" headline** (finding 23). README + positioning; small.

### Tier 2 — Exploit the hardware moat / extend differentiators

- **T2.1 FP8 compute path** (torchao / Transformer Engine), gated on Blackwell (finding 4). *Effort: M–L; experimental opt-in first.*
- **T2.2 SLAO → multi-strategy, eval-gated merge framework** (findings 30–31); reuse the v1.4 drift gate. *Effort: M.*
- **T2.3 Adapter-native export options** (`--format ollama-adapter`, "adapter shelf" hot-swap) + `use_rslora` validation (findings 19, 27–29). *Effort: M.*

### Tier 3 — New lanes (bigger lifts)

- **T3.1 MLX / Apple-Silicon backend** (`[mlx]` extra; retire the obsolete boundary; dogfood on the M5 Max) (findings 21–22). *Effort: L.*
- **T3.2 Reasoning-trace SFT recipe** (`<think>` chat-template handling + trace-length filtering) (finding 24); serves the studio's distillation needs. *Effort: M.*

---

## 4. Per-feature briefs (load-bearing items)

### T1.2 — ORPO (the highest-confidence feature bet)

- **What:** a `method="orpo"` (or `--method orpo`) training mode. ORPO = standard SFT NLL loss + a per-step odds-ratio penalty over (chosen, rejected) pairs. No reference model, single stage.
- **Why it fits:** TRL already implements `ORPOTrainer`; the work is dataset-format detection for preference pairs (chosen/rejected) + wiring the trainer + presets + docs — not a new training loop.
- **Scope discipline:** ship ORPO first (cheapest, single-stage). SimPO/KTO are documented follow-ons. **Do NOT** add PPO/GRPO. Reframe the README boundary to: *"single-stage SFT + reference-free preference tuning (ORPO/SimPO/KTO); no online RL (PPO/GRPO) — for those use TRL/LLaMA-Factory."*
- **Dataset contract:** auto-detect `{chosen, rejected}` / `{prompt, chosen, rejected}` shapes alongside the existing SFT formats.

### T1.1 — Dataset-quality + eval loop (the moat)

- **Quality report (`backprop data report <jsonl>`):** duplicate clusters, length/format outliers, train/test contamination flags, format-validity, token-length histogram — extends the existing `datasets.py` detect/dedupe/filter/curriculum.
- **Lightweight eval harness (`backprop eval <run-id>`):** held-out loss + N sample generations against a fixed prompt set, with a before/after diff for multi-run/continual campaigns. Gates the SLAO merges (ties to T2.2).
- **Why this and not breadth:** competitors compete on recipes/RL/model-count; none serve the solo prep→train→eval loop. On-brand, defensible, and it is the loop *we* run internally.

### T2.2 — SLAO merge framework + eval-gated merges

- **What:** make the multi-run merger pluggable — `merge_strategy ∈ {qiao_mahdavi (default), ties, dare, linear}` — add a similarity/drift gate (reuse v1.4 drift-gate-5-class) deciding merge-vs-branch, and an eval-gate (reuse T1.1 eval) that rejects a merge regressing the held-out set.
- **Claim hygiene:** keep "implements Qiao & Mahdavi 2025, first known downstream adopter"; drop any "best continual-learning method" framing (finding 30).

### T3.1 — MLX / Apple-Silicon backend

- **What:** an `[mlx]` extra + an MLX training backend behind the same `Trainer` API; `mlx_lm.lora` under the hood; `mlx_lm.fuse` → GGUF → Ollama for the export path.
- **Boundary update:** macOS `trainer.train()` currently raises `DEP_GPU_NOT_AVAILABLE`; under `[mlx]` it should route to the MLX backend instead. CUDA stays canonical; MLX is the second rail.
- **Dogfood:** validate on the M5 Max (128 GB unified) — a family the public also wants.

---

## 5. Out of scope for v1.5 (boundaries that HOLD, with evidence)

- **Online RL — PPO / GRPO / RLVR** (finding 14): genuinely heavyweight; cede to TRL/LLaMA-Factory.
- **Full-parameter fine-tuning of 7B+** : needs 24 GB+; `mode="full"` stays ≤3B.
- **Multi-node / distributed**: single machine only.
- **2-bit quantization (AQLM/QuIP#)**: breaks the merge/export contract (finding 1); revisit only if HQQ/torchao-int8 prove a mergeable sub-4-bit path (finding 6).
- **GaLore for fine-tuning** (finding 7): pretraining tool; Q-GaLore is a *future separate mode*.
- **Vision/multimodal LoRA** (finding 26): dilutes the 3-line promise on 16 GB; defer.

---

## 6. Sequencing into the swarm

This brief drives the **Feature Pass (Phases 5–8)**. It does NOT change the order of operations:

1. **Health Pass first** — Stage A (bug/security) → B (proactive) → C (humanization) → D (visual) on the clean v1.4.0 tree, to a clean bill of health. T0.1–T0.4 (doc/preset corrections) fold naturally into the CI/Docs lane here.
2. **Feature Pass** — execute Tier 1, then Tier 2, then Tier 3, user-reviewed each wave, exclusive file ownership, build-verified.
3. **Composed re-audit** — re-audit the composed surface (preference tuning × export × continual-merge seams) per the db-cluster close.
4. **Phase 9 final test → Phase 10 Full Treatment → ship v1.5.0.**

**Open questions for the director:**
- Tier ranking — is the dataset-quality/eval loop (T1.1) the right #1, or should ORPO (T1.2) lead?
- Is MLX (T3.1) in-scope for v1.5, or a v1.6 lane?
- Binaries: confirm `release-binaries.yml` (mac+win, Linux-excluded) should be committed to `main` as part of v1.5 — it matches the shipped npm-launcher architecture.
