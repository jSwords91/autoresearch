# autoresearch-compress

This is an experiment to have the LLM do its own research — but instead of
pretraining a model from scratch, the agent starts from an existing
pretrained checkpoint (`HuggingFaceTB/SmolLM2-360M-Instruct`) and tries to make it
smaller on disk while keeping its behavior essentially unchanged.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `compress/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b compress/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, model/data download, and the evaluation harness. Do not modify.
   - `compress.py` — the file you modify. Compression technique, optional recovery fine-tune, checkpoint saving.
4. **Verify the baseline exists**: Check that `~/.cache/autoresearch-compress/baseline.json` exists. If not, tell the human to run `uv run prepare.py` (downloads the model + eval data and computes baseline metrics, a few minutes).
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The script runs for a **fixed time
budget of 10 minutes** (wall clock, covering compression + optional recovery
fine-tune + checkpoint save + evaluation). You launch it simply as: `uv run
compress.py`.

**What you CAN do:**
- Modify `compress.py` — this is the only file you edit. Everything is fair game: post-training quantization (int8/int4, weight-only or activation-aware), structured or unstructured pruning, low-rank factorization (SVD decomposition of weight matrices), knowledge distillation to a smaller architecture, layer/head removal, weight sharing, mixed precision per layer, or any combination.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed model ID, eval datasets, quality-gate tolerances, and the evaluation functions themselves.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml` (`torch`, `transformers`, `accelerate`, `datasets`, `safetensors`, `bitsandbytes`, `huggingface-hub`, `numpy`).
- Change the tokenizer or vocabulary. The eval harness assumes the baseline tokenizer throughout — a technique that needs a different vocabulary would need its own eval path, which is out of scope here.
- Modify the evaluation harness or gate tolerances (`compute_bpb`, `compute_lambada_accuracy`, `run_generation_sanity`, `BPB_TOLERANCE`, `LAMBADA_TOLERANCE_ABS`, `GEN_SANITY_TOLERANCE_ABS` in `prepare.py`). These are the ground truth.

## The goal: quality gate + compression objective

Unlike a training loss that can improve indefinitely, compression has a
cliff — shrink too far and the model breaks in ways a single metric can
miss. So the goal here is **not** "minimize one number." It's a constrained
optimization with two parts:

1. **Hard quality gate** (must pass, no partial credit): the compressed
   checkpoint's metrics must stay within tolerance of the baseline on *all
   three* of:
   - `bpb_wikitext2` (bits-per-byte on held-out WikiText-2 — fluency/world-knowledge signal)
   - `lambada_acc` (LAMBADA cloze accuracy — downstream task signal)
   - `gen_sanity_pass_rate` (fixed instruction-prompt suite, checked for empty/degenerate/NaN output — catches coherence collapse that the other two can miss)

   These three are intentionally different axes so an experiment can't "win"
   by gaming one narrow metric. If ANY of them regresses beyond tolerance,
   the experiment is a `discard` — full stop, regardless of how much smaller
   the checkpoint got.

2. **Objective to maximize** (only among experiments that pass the gate):
   `compression_ratio` = baseline size in MB / compressed size in MB, where
   size is measured as actual bytes on disk of the saved checkpoint
   directory. Size-on-disk was chosen over parameter count because it's
   honest across every technique — quantization, structured pruning, and
   distillation-to-a-smaller-architecture all show up in it the same way.

**Important caveat**: unstructured pruning (zeroing individual weights)
does **not** reduce `size_mb` unless you also serialize the result in a
genuinely sparse format — a dense safetensors file full of zeros is the same
number of bytes. If you want credit for pruning, either use structured
pruning (drop whole heads/channels/layers so the saved tensors are actually
smaller) or combine unstructured pruning with a real sparse serialization
and confirm the on-disk size actually dropped before calling it a win.

**VRAM** (`peak_vram_mb`, printed but not gated) is a soft constraint. Some
increase during compression (e.g. for a recovery fine-tune) is fine; don't
let it blow up dramatically.

**Simplicity / trust tie-breaker**: when two techniques land at a similar
compression ratio, prefer the simpler and more standard one (e.g. plain
weight-only quantization over quantization + pruning + distillation
stacked together) — easier to trust, easier to reproduce, less likely to be
overfit to this particular eval suite.

**The first run**: your very first run should always be the baseline as-is
(`compress()` is a no-op by default) — this establishes the `compression_ratio
== 1.0` baseline row.

## Output format

Once the script finishes it prints a summary like this:

```
---
commit:            a1b2c3d
size_mb:           940.21
compression_ratio: 1.000
bpb_wikitext2:     0.812340
lambada_acc:       0.5133
gen_sanity_pass:   1.000
tokens_per_sec:    42.3
peak_vram_mb:      2103.4
quality_gate:      PASS
total_seconds:     187.4
```

Extract the key fields from the log file:

```
grep "^size_mb:\|^compression_ratio:\|^bpb_wikitext2:\|^lambada_acc:\|^gen_sanity_pass:\|^quality_gate:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT
comma-separated — commas break in descriptions).

The TSV has a header row and 10 columns:

```
commit	size_mb	compression_ratio	bpb_wikitext2	lambada_acc	gen_sanity_pass	tokens_per_sec	quality_gate	status	description
```

1. git commit hash (short, 7 chars)
2. size_mb of the saved checkpoint (e.g. 312.40) — use 0.0 for crashes
3. compression_ratio vs baseline (e.g. 3.012) — use 0.000 for crashes
4. bpb_wikitext2 achieved (e.g. 0.821450) — use 0.000000 for crashes
5. lambada_acc achieved (e.g. 0.4980) — use 0.0000 for crashes
6. gen_sanity_pass rate (e.g. 0.917) — use 0.000 for crashes
7. tokens_per_sec (e.g. 38.2) — use 0.0 for crashes
8. quality_gate: `PASS` or `FAIL` — use `FAIL` for crashes
9. status: `keep`, `discard`, or `crash`
10. short text description of what this experiment tried

Example:

```
commit	size_mb	compression_ratio	bpb_wikitext2	lambada_acc	gen_sanity_pass	tokens_per_sec	quality_gate	status	description
a1b2c3d	940.21	1.000	0.812340	0.5133	1.000	42.3	PASS	keep	baseline
b2c3d4e	480.10	1.958	0.815900	0.5100	1.000	61.7	PASS	keep	int8 weight-only quantization (bitsandbytes)
c3d4e5f	480.10	1.958	0.841200	0.4820	0.917	60.9	FAIL	discard	int4 quantization, no recovery finetune
d4e5f6g	0.00	0.000	0.000000	0.0000	0.000	0.0	FAIL	crash	structured head pruning (shape mismatch in attention)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `compress/mar5` or
`compress/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on, and the best `compression_ratio` kept so far.
2. Tune `compress.py` with an experimental idea by directly hacking the code.
3. git commit.
4. Run the experiment: `uv run compress.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context).
5. Read out the results: `grep "^quality_gate:\|^compression_ratio:" run.log`.
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on this idea.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git).
8. Decide keep/discard:
   - `quality_gate: FAIL` → always `discard`, regardless of compression_ratio.
   - `quality_gate: PASS` and `compression_ratio` beats the best kept so far → `keep`, advance the branch (git commit stays).
   - `quality_gate: PASS` but `compression_ratio` doesn't beat the best kept so far → `discard` (it's a valid but non-improving result).
9. If `discard`, `git reset` back to where you started before trying the next idea.

The idea is that you are a completely autonomous researcher trying things
out. If they work, keep. If they don't, discard. And you're advancing the
branch so that you can iterate. If you feel like you're getting stuck in
some way, you can rewind but you should probably do this very very sparingly
(if ever).

**Timeout**: Each experiment should take ~10 minutes total (+ a few seconds
for startup overhead). If a run exceeds 20 minutes, kill it and treat it as
a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment:
If it's something dumb and easy to fix (e.g. a typo, a shape mismatch), fix
it and re-run. If the idea itself is fundamentally broken, just skip it, log
"crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial
setup), do NOT pause to ask the human if you should continue. Do NOT ask
"should I keep going?" or "is this a good stopping point?". The human might
be asleep, or gone from a computer and expects you to continue working
*indefinitely* until you are manually stopped. You are autonomous. If you
run out of ideas, think harder: try combining previous near-misses (e.g.
quantization + a short recovery fine-tune), try more aggressive
quantization on some layers and less on others (embeddings and the final
layer are usually more sensitive than middle MLP blocks), try structured
pruning at different granularities (attention heads, whole layers, MLP
intermediate channels), try distilling to a shallower/narrower student
initialized from the teacher's own layers. The loop runs until the human
interrupts you, period.

As an example use case, a user might leave you running while they sleep. If
each experiment takes you ~10 minutes then you can run approx 6/hour, for a
total of about 50 over the duration of the average human sleep. The user
then wakes up to experimental results, all completed by you while they
slept!
