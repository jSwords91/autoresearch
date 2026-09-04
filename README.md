# autoresearch-compress

A fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch),
pointed at a different problem: instead of an AI agent pretraining a small
GPT from scratch overnight, it takes an existing pretrained model
(`HuggingFaceTB/SmolLM2-360M-Instruct`) and tries to compress it — quantization,
pruning, low-rank factorization, distillation, or any combination — while
keeping its behavior essentially unchanged.

The idea: give an AI agent a small but real pretrained model and a fixed
time budget per experiment, and let it try compression techniques
autonomously overnight. It modifies the code, runs a compression pass, checks
whether the result still behaves like the original within tolerance, keeps
or discards, and repeats. You wake up in the morning to a log of experiments
and (hopefully) a substantially smaller model that still works.

Same division of labor as the original: you don't touch the Python files
like you normally would as a researcher — you program `program.md`, a
Markdown file that gives the agent its instructions and its autonomous
research org. The Python files are the fixed harness plus the one file the
agent iterates on.

## Why this is harder than "get a lower loss"

In the original repo, training loss can improve indefinitely: lower
`val_bpb` is always better, so the whole loop is "minimize one number within
a fixed time budget." Compression doesn't work that way. Shrink a model too
far and it breaks, often in ways a single narrow metric won't catch.

The key insight this fork is built on: **compression has a privileged
reference that ordinary model evaluation does not - the original model.**
The question isn't "is the compressed model good", it's "is it the same as
this specific frozen artifact". Absolute metrics like perplexity conflate
the original's own errors with the damage you caused; agreement with the
original isolates the damage. So nearly every metric here is comparative,
measured at three levels of strictness:

- **`top1_agreement`** - does it emit the same token? This is what a user
  actually sees under greedy decoding.
- **`kl_div`** - does it hold the same distribution? Catches confidence loss
  that top-1 hides. Empirically the most sensitive instrument available: a
  2x change in quantization bit width moves it 5.5x.
- **`gen_agreement`** - does it still say the same thing when running free?
  The other two are teacher-forced and blind to compounding error.

That reframing matters. Under an earlier absolute-metric harness, 8-bit
quantization looked like a clean pass. Measured against the original it
changes the emitted token **1 time in 11**.

Two further design commitments:

- **The gate detects breakage; the frontier measures value.** Conflating
  those produced 17 undifferentiated `FAIL` rows in one session, with no way
  to rank them. Now a loose gate asks only "is this broken", and surviving
  experiments are ranked by Pareto dominance over compression, fidelity, and
  speed.
- **Speed is gated, not just logged.** An objective that counts only bytes
  will happily buy size with latency: the 8-bit checkpoint above is 1.76x
  smaller and 4.7x *slower*.

Full rationale, calibration reference points, and thresholds are in
`program.md`.

## How it works

- **`prepare.py`** — fixed constants, one-time setup (downloads the
  baseline model + eval data), and the evaluation harness (agreement with
  the frozen original, generation sanity, size, throughput). Not modified.
- **`compress.py`** — the single file the agent edits. Contains the
  compression technique, an optional recovery-fine-tune helper, and the
  experiment runner. Everything about *how* you compress the model is fair
  game. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent
  here and let it go. **This file is edited and iterated on by the human**.

By design, each experiment runs for a **fixed 10-minute time budget** (wall
clock, covering compression + optional recovery + eval), regardless of the
details of your compute — same rationale as upstream: comparable
experiments regardless of what the agent changes, and it finds the best
result your platform's time budget allows.

## Quick start

**Requirements:** A single NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download the baseline model + eval data, compute baseline metrics (one-time, a few minutes)
uv run prepare.py

# 4. Manually run a single compression experiment (~10 min, no-op by default)
uv run compress.py
```

If the above commands all work ok, your setup is working and you can go
into autonomous research mode.

## Running the agent

Spin up your Claude/Codex or whatever you want in this repo (and disable
all permissions), then prompt something like:

```
Hi, have a look at program.md and let's kick off a new compression experiment! Let's do the setup first.
```

## Project structure

```
prepare.py      — constants, model/data download, evaluation harness (do not modify)
compress.py     — compression technique, recovery fine-tune, experiment runner (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `compress.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Every experiment runs for exactly 10 minutes, regardless of your specific platform.
- **Measure agreement with the original, not absolute quality.** Absolute metrics conflate the original model's own errors with compression damage. The original is a fixed reference, so comparing against it isolates exactly what you broke, and does so far more densely: ~19k token positions of full-distribution comparison per run, versus a few hundred binary accuracy outcomes.
- **Separate "is it broken" from "is it good value".** A single tight bar answers neither well. A loose gate catches breakage; Pareto ranking over compression, fidelity and speed handles value.
- **Size-on-disk as the compression metric.** Robust across techniques; also makes "did this actually compress anything" a literal, checkable fact rather than a claim.

## License

MIT (inherited from upstream). Model weights (`HuggingFaceTB/SmolLM2-360M-Instruct`) are Apache 2.0, licensed separately by Hugging Face (SmolLM2).
