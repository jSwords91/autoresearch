# autoresearch-compress

A fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch),
pointed at a different problem: instead of an AI agent pretraining a small
GPT from scratch overnight, it takes an existing pretrained model
(`Qwen/Qwen2.5-0.5B-Instruct`) and tries to compress it — quantization,
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

In the original repo, training loss can improve indefinitely — lower
`val_bpb` is always better, so the whole loop is "minimize one number
within a fixed time budget." Compression doesn't work that way: shrink a
model too far and it breaks, often in ways a single narrow metric won't
catch. So this fork reframes the loop as a constrained optimization —

- **Hard quality gate** (must pass): the compressed model must stay within
  tolerance of the baseline on three different signals — bits-per-byte on
  held-out WikiText-2 (fluency), LAMBADA cloze accuracy (downstream task),
  and a generation-sanity check on a fixed instruction-prompt suite
  (catches coherence collapse that aggregate metrics can miss). An
  experiment that regresses on any of these is a `discard`, no matter how
  much smaller it got.
- **Objective to maximize** (only among experiments that pass the gate):
  compression ratio, measured as actual bytes on disk — honest across
  quantization, pruning, and distillation alike, unlike raw parameter count.

Full rationale and the exact tolerances are in `program.md`.

## How it works

- **`prepare.py`** — fixed constants, one-time setup (downloads the
  baseline model + eval data), and the evaluation harness (bpb, LAMBADA
  accuracy, generation sanity, size, latency). Not modified.
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
- **Three-signal quality gate, not one.** A single perplexity number is easy to game inadvertently (a technique can hold aggregate loss steady while wrecking a specific capability). Bpb + LAMBADA + generation sanity are different enough that this is hard to do by accident.
- **Size-on-disk as the compression metric.** Robust across techniques; also makes "did this actually compress anything" a literal, checkable fact rather than a claim.

## License

MIT (inherited from upstream). Model weights (`Qwen/Qwen2.5-0.5B-Instruct`) are Apache 2.0, licensed separately by Alibaba/Qwen.
