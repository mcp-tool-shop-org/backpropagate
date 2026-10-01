# E4: where does full fine-tuning beat QLoRA? Pre-registration (code task)

**Status: pre-registered. No run exists at the commit that adds this file.** Everything below the
heading "Pre-registered gates" was written before any model was trained or evaluated for E4, and
the numbers in it are not to be tuned afterwards. Results will be added next to this file as
`stage_e4_3b.json`, `stage_e4_7b.json`, `runs/`, `budget_*.json` and `precheck_*.json`.

E4 decides whether the experimental block-coordinate engine (engine B, `--full-ft-engine block`)
has a future. Background and the earlier test: `docs/handoff-2026-09-30-full-ft-experiments.md`
(sections 3.3, 4/E4, 5, 6) and `docs/receipts/2026-09-30-gsm8k/` (stage d).

## Why stage d could not answer the question, and what E4 changes

| Stage d problem | E4 change |
|---|---|
| `Qwen2.5-*-Instruct` were already tuned on this kind of data; training made every arm worse than the untrained model | **Base** checkpoints: `Qwen/Qwen2.5-3B` and `Qwen/Qwen2.5-7B` (Biderman et al. 2024, arXiv:2405.09673, also used base models) |
| About 2M tokens per arm (1000 steps x 4 x 512), far below where Biderman's full-FT-vs-LoRA gap appeared | **5000 steps x batch 4 x 512 tokens** (10.24M token slots, about 8.3M real tokens, see below), same lr / schedule / optimizer per arm as stage d |
| Task the models had effectively seen | A code task whose data was created after Qwen2.5 was released, evaluated on held-out problems from the same source |

## Step zero: dataset licence check (done 2026-09-30, every claim read on the Hub)

Raw evidence (licence strings, Hub shas, card URLs, generator statements): `licence_evidence.json`.

| Candidate | Licence (card) | Generator model(s) | Rows | Unit tests per row | Verdict |
|---|---|---|---|---|---|
| **`nvidia/OpenCodeInstruct`** | CC BY 4.0 | Qwen2.5-32B-Instruct and Qwen2.5-Coder-32B-Instruct (both Apache-2.0), per arXiv:2504.04030 | 5,000,000 | yes (10 per row) plus execution status | **accepted**, `domain == generic` rows with every test passing |
| `bigcode/self-oss-instruct-sc2-exec-filter-50k` | odc-by | StarCoder2-15B | 50,661 | **no** (tests were used to filter, not stored) | rejected here: pass@1 cannot be executed against it |
| `OpenCoder-LLM/opc-sft-stage2` | MIT | **card names none**; its `evol_instruct` part is MagicCoder-Evol-Instruct-110k; the paper says a related seed set was synthesized with GPT-4o-Mini | 118,278 (educational) | partly | rejected: provenance unclear |
| `ise-uiuc/Magicoder-OSS-Instruct-75K` | MIT | **gpt-3.5-turbo-1106 (OpenAI)**, stated on the card | about 75K | no | rejected: OpenAI-generated |

Why OpenCodeInstruct passes: CC BY 4.0 (commercial use allowed, attribution required: credit
NVIDIA and arXiv:2504.04030 wherever results are published); the paper names only Qwen models for
the steps it names and has **zero** mentions of OpenAI, Anthropic, Gemini or Claude models (the
word GPT occurs only in reference titles). Hub sha read: `8f3ba5bafe4d6e8db46082cf7ae6741bc370604d`
(2026-09-30); the driver downloads that exact revision, not `main`.

**Residual provenance risk, for the Director to accept or refuse.** The OpenCodeInstruct paper
says the Genetic-Instruct instruction-evolution step used "an LLM" and does not name it (the
framework paper, arXiv:2407.21077, used Mixtral-8x22B and Qwen models). Every model the papers
do name is open-weight with a permissive licence, and no proprietary model is named anywhere, but
that one step's exact model is not documented. Only the `generic` domain is used; the
TACO-seeded `algorithmic` rows (competitive-programming questions, mixed provenance) are excluded.

Also recorded: the **generator is the same model family as the model being trained** (Qwen
teacher, Qwen student). That makes the data easy to learn for every arm; it is a property of the
test, not a bias between arms.

**Base-model licences (read on the Hub):** `Qwen/Qwen2.5-7B` is Apache-2.0 (sha `d149729398`).
`Qwen/Qwen2.5-3B` is the **Qwen Research Licence** (`license: other`, sha `3aab1f1954`; the
licence defines non-commercial as "research or evaluation purposes only"). E4 is an evaluation;
the 3B weights it produces must not ship.

## Evaluation design

Primary metric: **held-out loss on the reference solutions, answer tokens only** (prompt tokens
masked), token-weighted over the 500 eval problems, truncated at `--seq` tokens exactly like
training, as in stage d. Also recorded as a secondary field: the same loss with sequences up to
1024 tokens (`answer_loss_long`), because 27% of eval texts exceed 512 tokens.

Secondary metric: **pass@1 with greedy decoding, executed against the problem's unit tests.** A
generation passes when the first Python code block of the reply passes every unit test of its
problem (the dataset provides about 10 per problem; rows where the reference solution itself
fails any of them are excluded, so a failure is the model's, not a broken test).
Generation: prompt ends at `<|im_start|>assistant\n`, up to 768 new tokens (reference answers
reach 658 tokens, p99 601), stop on `<|im_end|>` / eos. A generation that hits the limit is
usually unparseable and counts as a failure; the count is recorded (`hit_max_new_tokens`).

Why this eval set rather than HumanEval / MBPP: OpenCodeInstruct was created January-March 2025
(card), **after** Qwen2.5's release (September 2024), so its problems cannot be in the base
model's pretraining; the HumanEval / MBPP families (and their EvalPlus extensions) are widely
reproduced on the web and may be. The eval problems are held out from training by a seeded split
and by normalised-hash deduplication. The cost: pass@1 here measures in-distribution competence
at the dataset's own task style, not a public benchmark number; that is the right yardstick for
"does full fine-tuning beat QLoRA at fitting this data" and the wrong one for "is the model a
better coder in general". HumanEval+/MBPP+ are not run.

Interface hint. Only about half of the dataset's questions name the function the tests call, so
a model could not pass by skill alone. Every prompt (train and eval alike) therefore ends with
the signatures of the top-level names that the unit tests call, taken from the reference
solution, and a one-line instruction to reply with one Python code block. Same format in
training and evaluation; rows whose tests call nothing the reference defines are dropped (232 of
100,000).

### The data (`pre/prep_code.json`, produced by the same driver the pod runs; shard 00000 of 00050)

| Step | Rows |
|---|---|
| Raw rows in the shard | 100,000 |
| Dropped: `domain == algorithmic` | 9,129 |
| Dropped: reference not passing every generated test (`average_test_score < 1`) | 60,519 |
| Dropped: tests call nothing the reference defines | 232 |
| Valid pool | 30,120 |
| Eval split (seed 0, shuffled) | 500 |
| Train candidates removed: overlap with eval by normalised hash | **27** (1 by prompt, 26 by solution) |
| Train candidates removed: repeats within train | 437 |
| **Train** | **29,156** |

Dedupe key: SHA-256 of the lower-cased, whitespace-collapsed question text, and separately of the
reference code block. 5000 steps x batch 4 = 20,000 examples, so training is 0.69 of an epoch with
no repeated example, in the same order for every arm with the same seed.

### Token budget and truncation (`pre/lengths_code_Qwen_Qwen2.5-3B.json`, tokenizer only; Qwen2.5 3B and 7B share one vocabulary)

| | 4 x 512 (default) | 2 x 1024 (proposed alternative) | 4 x 768 (extra option) |
|---|---|---|---|
| Train examples cut (prompt + solution over the limit) | **29.0%** | 1.3% | 6.3% |
| Real (non-padding) tokens per step | 1,664 | 925 | about 1,818 |
| Real tokens at 5000 steps | **8.3M** | 4.6M | about 9.1M |
| Steps to reach 8.3M real tokens | 5000 | about 9000 | about 4600 |
| Measured s/step (stage d) | yes | no | no |

The pre-registration says: if more than about 5% of reference solutions are truncated, propose
batch 2 x 1024 rather than choosing it silently. **29% are truncated at 512, so this is the
proposal, and it is not applied: the defaults stay 5000 x 4 x 512**, pending the lead's decision.
Two things the lead should weigh. (1) Truncation is identical in every arm, so the comparison is
fair, but about 29% of examples never show the model the end of their solution, and the primary
loss is truncated the same way. (2) 2 x 1024 at 5000 steps trains on only 55% as many real tokens
as 4 x 512 and its step time is unmeasured; the 7B stage is fixed at 4 x 512 anyway (engine B
peaks at 30.0 GiB NVML at that shape, no context scaling). Override with `E4_BATCH` / `E4_SEQ`.

## Pre-registered gates

The first five bullets are the pre-registration as given, **verbatim**.

- **Pre-check (abort gate):** untrained base model, on the eval set: pass@1 must be strictly inside [0.10, 0.80] and held-out loss recorded. Outside -> abort before training; the run spends nothing more.
- **Stage 3B:** standard full FT, QLoRA r=256, engine B K=5 - 3 seeds each (9 runs), 5000 steps.
- **Premise gate:** full FT beats QLoRA at 3B if held-out loss is >=0.05 nats lower, or pass@1 is higher with McNemar p<0.05 - the same rule and statistics as stage d (pooled seeds, paired bootstrap 10k, McNemar exact). **If it fails, stop: the 7B stage does not run.**
- **Stage 7B (only if the premise holds):** engine B K=5 and QLoRA r=256 x 2 seeds, GaLore layerwise 8-bit x 1 seed (same GaLore args as stage d), batch 4 x 512 only (engine B peaks at 30.0 GiB NVML there - no context scaling).
- **Ship rule:** engine B leaves experimental only if it beats QLoRA at 7B under the same rule. GaLore is reported with the same comparisons; adopting it is the Director's call.
- **Budget guard:** pod caps are $4.00 (3B stage) and $4.50 (7B stage) at ~$0.90/h.

### How the verbatim text is computed (`scripts/e4_lib.py`, pure functions with tests)

Where the text left a choice, this is the choice. These interpretations are part of the
pre-registration; the lead reviews them before the pod exists.

1. **Pre-check:** `precheck_verdict` requires `0.10 < pass@1 < 0.80` (both ends excluded) and a
   finite held-out loss. It is run once per model before that model's stage: 3B before the 3B
   stage, 7B before the 7B stage. A failing 3B pre-check ends E4 with no training spend. An
   aborted pre-check whose sampled generations look sane but fail the pass rate for format
   reasons (a base model asked in ChatML) is a Director question, not something to work around.
2. **"Full FT"** means the library's standard pure-GPU full fine-tuning (`default` arm, 8-bit
   paged AdamW, lr 2e-5). **QLoRA** is r=256 at the library default lr 2e-4. Engine B is
   `--full-ft-engine block`, K=5, lr 2e-5. All arms: batch 4, no gradient accumulation, no
   packing, cosine schedule, 10 warmup steps, weight decay 0.01, full-sequence training loss, the
   same data order per seed (checked per seed from the first batch's hash and loss mask).
   GaLore: `galore_adamw_8bit_layerwise`, `rank=128, update_proj_gap=200, scale=0.25`, ceiling
   override 8, as in stage d.
3. **Loss criterion:** the pooled (per item, mean over seeds, token-weighted) held-out-loss
   difference `loss(A) - loss(B)` is **at most -0.05 nats**. The point estimate decides, as in
   stage d; the 95% bootstrap interval (10,000 resamples of items, seed 0) is reported alongside
   with a flag for whether it excludes zero.
4. **Accuracy criterion:** an item counts as solved by an arm if it is solved in strictly more
   than half of that arm's seeds (3 seeds: at least 2; 2 seeds: both). McNemar exact (two-sided)
   on those binary vectors must give `p < 0.05` **and** more items solved only by A than only by
   B. Per-seed McNemar tests (seed-to-seed pairs) and the pooled pass@1 difference with its
   bootstrap interval are reported but do not decide.
5. **Seeds:** a gate needs at least 2 completed seeds in each arm of the pair, otherwise the
   verdict is `INCONCLUSIVE`, which, like `FAIL`, **stops** the 3B-to-7B step. The 7B stage
   refuses to start without a `PASS` in `stage_e4_3b.json` (an explicit override must be named in
   the environment and is written into the receipt).
6. **Pairs that decide:** premise = `default` vs `qlora` at 3B. Ship rule = `block_k5` vs `qlora`
   at 7B. Everything else (engine B vs full FT / QLoRA at 3B, GaLore vs QLoRA and vs engine B at
   7B) is reported with the same statistics and decides nothing.
7. **A failed or out-of-memory run** is a recorded result, not a retry; its seed is simply absent.

## Run plan, budget guard and drop order

Each stage is an ordered list. The guard (`BudgetGuard`) admits a run only if its estimated end
is before `start + cap / $0.90 per h - 10 min`; **the first run that does not fit is dropped and
so is everything after it** (a cheap run never jumps ahead of a dropped higher-priority one).
The estimate for an arm is replaced by its measured wall time as soon as one run of that arm has
finished; the first estimate uses the stage-d s/step and the measured eval time of the base model.

| Stage | Run order (earlier = higher priority) | Drop order (first to go) |
|---|---|---|
| 3B | default s0, qlora s0, default s1, qlora s1, default s2, qlora s2, block_k5 s0, s1, s2 | block_k5 s2, s1, s0, then qlora s2, default s2, qlora s1, ... |
| 7B | block_k5 s0, qlora s0, block_k5 s1, qlora s1, galore s0 | galore, then qlora s1, block_k5 s1, ... |

Rationale: the premise gate needs full FT and QLoRA, so those come first; engine B at 3B is
context for the 7B decision and is the cheapest to lose. At 7B the ship rule needs engine B and
QLoRA; GaLore costs the most and its adoption is a separate decision.

### Estimated pod time and cost (from stage d's measured s/step; eval times are estimates, no code eval has run yet)

Training only: 3B default 26.4 min, QLoRA 36.2 min, engine B 12.0 min per run of 5000 steps; 7B
engine B 18.3 min, QLoRA 44.5 min, GaLore 105.9 min. Eval is estimated at 4.5 min (full FT /
engine B at 3B), 9 min (QLoRA at 3B), 7 and 14 min at 7B (generation of 500 answers of up to 768
tokens plus the sandboxed tests; QLoRA is slower because its adapter is not merged, as in stage d).

| Stage | All planned runs | Setup (install, model download, prep, base eval) | Total at $0.90/h | Cap | What the guard admits |
|---|---|---|---|---|---|
| 3B (9 runs) | 4.85 h | about 0.25 h | **5.1 h, $4.6** | $4.00 | default x3, qlora x2; **engine B x3 and qlora s2 dropped** |
| 7B (5 runs) | 4.88 h | about 0.35 h | **5.2 h, $4.7** | $4.50 | block_k5 x2, qlora x2; **GaLore dropped** |

**The full pre-registered plan does not fit the caps.** Options for the lead (none is applied):
(a) raise the 3B cap to about $4.80 and the 7B cap to about $5.30 (env `E4_BUDGET_USD_3B` /
`_7B`); (b) run 2 seeds first (`E4_SEEDS_3B="0 1"`, about $3.1) and add seed 2 and engine B only
if the gate is borderline (stages resume: a run with a receipt is skipped); (c) accept the
guard's drops (the premise gate still has 3 vs 2 seeds). The numbers are in the table because the
brief asked for them; they come from `scripts/e4_lib.py` (`estimate_run_s`, `walk_plan`).

Disk: the model cache on the container disk (about 6.2 GB for 3B, 15.2 GB for 7B, which the
7B pre-check purges the 3B copy before downloading), the repo and Python packages (about 5 GB),
per-run scratch (the end-of-run checkpoint is deleted; no checkpoints are written during the
run) and receipts (about 6 MB, per-item outputs are gzipped). A **50 GB container disk** is
comfortable; receipts must not be on the network volume's critical path.

## Exact pod command sequence

```
# on the pod, after pod.sh verify and arming the dead-man timer (the lead does both)
export E4_START_EPOCH=<unix time the pod started billing>   # optional; default: first command
tmux new -s e4
bash scripts/pod_e4.sh setup          # clone branch, install, sandbox self-test, dataset prep + length stats
bash scripts/pod_e4.sh precheck 3b    # base-model gate: exit 4 = abort, nothing is trained
bash scripts/pod_e4.sh stage 3b       # runs in plan order under the budget guard, provisional verdict after each run
bash scripts/pod_e4.sh gate 3b        # statistics + verdict -> stage_e4_3b.json; exit 0 = PASS, 3 = stop (FAIL/INCONCLUSIVE)
# scp $WORK/{runs,stage_e4_3b.json,budget_3b.json,precheck_3b.json,pre-stage files} back; only if the gate exited 0:
bash scripts/pod_e4.sh precheck 7b    # purges the 3B cache, downloads 7B, base-model gate
bash scripts/pod_e4.sh stage 7b
bash scripts/pod_e4.sh gate 7b        # ship rule -> stage_e4_7b.json
# scp receipts back, then the lead deletes the pod
```

Smoke before paying: `python scripts/pod_e4.py dry-run --out <dir>` runs the whole pipeline
(prep, base eval, both stages' arms, summaries, verdict computation) on CPU with
`HuggingFaceTB/SmolLM2-135M` for 3 steps and executes no generated code. `--model <local dir>`
swaps in another tiny model. `bash scripts/pod_e4.sh setup` also runs the executor's self-test
on the pod before anything is paid for.

## Sandboxing

Generated code executes **only on the pod**: `run_candidate` refuses unless `--allow-exec` is
passed on a POSIX host with `resource` (Windows always refuses), and the dry run never passes it.
Each candidate runs in a fresh interpreter (`python -I`) in its own session and temp directory,
with limits on CPU time, address space (2 GiB), file size and open files, no core dumps, sockets
disabled in the interpreter, stdin/stdout/stderr discarded (a candidate cannot forge its result),
a minimal environment, and a wall-clock timeout (10 s) that kills the whole process group.
Outcomes recorded per item: `pass`, `fail`, `load_error`, `timeout`, `crash`, `no_code`. This is
process-level isolation for a throwaway pod, not a security boundary against a hostile program.

## Known limits (stated now, not discovered later)

- pass@1 is in-distribution to the dataset's style, with a teacher of the same family as the
  student; it is not a public benchmark number.
- The interface hint makes tests passable but also tells the model the signature; every arm gets
  the same help.
- 29% of training examples are truncated at 512 tokens (see the proposal above).
- The QLoRA arm generates through the unmerged adapter on a 4-bit base, as in stage d.
- Qwen2.5-3B is research-licensed; its derived weights are not for release.
- One dataset shard (30K valid rows) is used; training sees 20,000 of them once.

## Standards compliance

| Standard | Score | Evidence |
|---|---|---|
| PIN_PER_STEP | 2 | Dataset revision, split seed, model ids, git SHA, versions, seeds and per-arm settings are pinned in code and written into every receipt; a run is replayable from the receipt. |
| ANDON_AUTHORITY | 3 | Sandbox self-test aborts `setup`; a failed pre-check exits 4 and `stage` refuses to train without it; a non-PASS premise verdict blocks the 7B stage; tests cover each gate. |
| NAMED_COMPENSATORS | 2 | Table below; the pod is created and deleted by the lead only. |
| DECOMPOSE_BY_SECRETS | 2 | Pure decision logic (`e4_lib.py`) is separate from GPU work (`pod_block_engine.py`), orchestration (`pod_e4.py`) and statistics (`pod_e4_summary.py`). |
| UNCERTAINTY_GATED_HUMANS | 2 | The lead decides the truncation proposal, the caps and the residual provenance question before the pod exists; a mixed gate result is reported, not auto-resolved. |
| EXTERNAL_VERIFIER | n/a | Not specialized work in the sense of the standing rule; licence terms were read from the Hub and papers directly. |

| Irreversible action | Undo | State after | Owner |
|---|---|---|---|
| Create a RunPod pod (billing starts) | `scripts/runpod/pod.sh delete <id>`; the dead-man timer deletes at the deadline | no pod, billing stopped | lead |
| Execute generated code | none needed: throwaway pod, rlimits, temp dirs; delete the pod | no residue | lead |
| Commit receipts to `main` | `git revert -m 1 <merge sha>` via a PR | `main` as before | lead |

## Files

| Path | Contents |
|---|---|
| `licence_evidence.json` | Licence strings, Hub shas, card URLs, generator statements for the candidates and models |
| `pre/prep_code.json` | The dataset split report, produced by the pod driver locally (no GPU), file hashes, eval ids |
| `pre/lengths_code_Qwen_Qwen2.5-3B.json` | Tokenizer-only length and truncation statistics |
| `precheck_*.json`, `runs/`, `stage_e4_*.json`, `budget_*.json` | Added after the run |
| `scripts/e4_lib.py` | Dataset prep, sandbox, statistics, decision rules, budget guard |
| `scripts/pod_block_engine.py` | Driver; `--dataset code` adds prep, lengths, base and train with the code eval |
| `scripts/pod_e4.py`, `scripts/pod_e4.sh`, `scripts/pod_e4_summary.py` | Orchestrator, pod entry point, stage summary |
| `tests/test_e4_harness.py` | CPU tests for all of the above |
