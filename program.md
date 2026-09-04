# autoresearch-compress

This is an experiment to have the LLM do its own research - but instead of
pretraining a model from scratch, the agent starts from an existing
pretrained checkpoint (`HuggingFaceTB/SmolLM2-360M-Instruct`) and tries to
make it smaller on disk while keeping its behaviour essentially unchanged.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `compress/<tag>` must not already exist - this is a fresh run.
2. **Create the branch**: `git checkout -b compress/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` - repository context.
   - `prepare.py` - fixed constants, model/data download, and the evaluation harness. Do not modify.
   - `compress.py` - the file you modify. Compression technique, optional recovery fine-tune, checkpoint saving.
4. **Verify the baseline exists**: Check that `~/.cache/autoresearch-compress/baseline.json` exists. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU with a **fixed time budget of 10
minutes** (wall clock, covering compression + optional recovery + checkpoint
save + evaluation). You launch it as: `uv run compress.py`.

**What you CAN do:**
- Modify `compress.py` - the only file you edit. Everything is fair game: post-training quantization (int8/int4, weight-only or activation-aware), structured or unstructured pruning, low-rank factorization, knowledge distillation, layer/head removal, weight sharing, mixed precision per layer, or any combination.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only: the fixed model ID, eval corpora, gate thresholds, and the metric definitions themselves.
- Install new packages. Use what is in `pyproject.toml` (`torch`, `transformers`, `accelerate`, `datasets`, `safetensors`, `bitsandbytes`, `huggingface-hub`, `numpy`). Referencing a different HuggingFace *dataset* is fine; adding a *package* is not.
- Change the tokenizer or vocabulary. The eval assumes the baseline tokenizer throughout.
- Train on the eval corpora. Fidelity uses the **test** splits of `no_robots` and `wikitext-2`; recovery fine-tuning must use train splits (e.g. `wikitext-103` train, `no_robots` train).

## How this is evaluated, and why

Compression has a privileged reference that ordinary model evaluation does
not: **the original model**. The question is not "is the compressed model
good", it is "is it the same as this specific frozen artifact". Absolute
quality metrics (perplexity, benchmark accuracy) conflate the teacher's own
errors with the damage you caused. Agreement with the teacher isolates the
damage. That is why nearly every metric here is comparative.

Agreement is measured at three levels of strictness, because there are three
distinct ways a compressed model can be wrong:

- **`top1_agreement`** - fraction of teacher-forced positions where your
  model's argmax matches the teacher's. This is the behavioural metric: it is
  exactly what determines greedy-decode output. Measured over ~19k token
  positions of mixed chat and prose, so it is dense and low-noise.
- **`kl_div`** - mean forward KL(teacher || student), nats/token. Catches
  what top-1 hides: the model still picks the right token but has lost
  confidence, which shows up under sampling and compounds in long
  generations. **This is the single most sensitive instrument here** - a 2x
  change in quantization bit width moves KL by 5.5x while moving top-1 by
  only 0.11. Watch it first.
- **`gen_agreement`** - free-running greedy generation compared against the
  teacher's, scored as mean normalised prefix length before divergence. The
  two metrics above are teacher-forced and therefore blind to compounding
  error; a model can match at 99% of forced positions and still spiral once
  it consumes its own output.

Plus one absolute check and two cost axes:

- **`gen_sanity_pass_rate`** - degeneration detector (empty / repetition
  loop / crash) on a fixed instruction suite. This is what catches the
  classic quantization failure, and `compress.py` prints the offending
  prompt and completion so you can see what actually broke.
- **`size_mb`** - actual bytes on disk. Honest across quantization, pruning
  and distillation alike, unlike parameter count.
- **`speed_ratio`** - throughput against a reference copy of the baseline
  measured **in the same run**, seconds apart. Do not compare against a
  cached figure: this GPU thermally throttles from 2100MHz to 210MHz over a
  session, so absolute tok/s drifts by 2.5x and only a same-run ratio is
  meaningful.

`lambada_acc` is printed for human legibility but is **not** gated. With a
few hundred binary outcomes it cannot resolve what we care about - in
calibration the 8-bit model scored *above* the unmodified baseline (0.44 vs
0.42), which is noise, not improvement.

## The gate detects breakage. The frontier measures value.

These are different questions and conflating them is a trap. A previous
iteration of this harness used one tight bar for both and produced 17
undifferentiated `FAIL` rows in a single session, which gave the search
nothing to climb.

**The gate** (`passes_quality_gate`) is deliberately loose and only asks *is
this model broken?*: no degeneration regression, KL below a catastrophic-
damage backstop, and throughput at least 0.8x. Passing means "not broken",
not "good".

**The frontier** ranks non-broken experiments on three higher-is-better axes
that are deliberately *not* collapsed into one scalar, because that would
bake in an exchange rate nobody has justified:

    compression_ratio    what you are buying
    top1_agreement       what you pay in fidelity
    speed_ratio          what you pay in latency

Use `dominates()`: keep an experiment if no previously-kept experiment beats
it on all three.

### Calibration reference points

Measured on this model, so you know what the numbers mean:

    technique   compress  top1    kl      gen_agr  sanity  speed   verdict
    8-bit bnb   1.757x    0.9116  0.0277  0.6867   1.000   0.213   FAIL: 4.7x too slow
    4-bit NF4   2.643x    0.8055  0.1516  0.5339   0.917   0.882   FAIL: repetition loop

Read those rows carefully, because they set the agenda. **4-bit is both
smaller and 4x faster than 8-bit**; its only defect is one repetition loop.
Fixing that defect is the most concrete open target in this project. 8-bit's
problem is structural: `LLM.int8()` dequantizes on every matmul, so it is
slow by construction at batch size 1 and no amount of tuning will fix it.

**Important caveat**: unstructured pruning (zeroing individual weights) does
**not** reduce `size_mb` unless you serialize in a genuinely sparse format -
a dense safetensors file full of zeros is the same number of bytes. Use
structured pruning, or confirm the on-disk size actually dropped.

### Mind the token budget

Recovery fine-tuning is the obvious way to repair a lossy compression, and
it is easy to fool yourself. The loop in `compress.py` is batch-size 1, so
four minutes is on the order of **100k tokens**. SmolLM2 was pretrained on
~4 trillion. A previous session burned five experiments concluding "recovery
does not work" when the real finding was that it never trained at all, at
1e-8 of the original budget.

So: either keep the damage small enough that recovery is a nudge rather than
a rebuild, or raise batch size / pack sequences so the token count reaches
the millions. And prefer distilling against the teacher's distribution (KL)
over LM loss on raw text - it is a strictly richer signal per token, which
is exactly what you need when tokens are the scarce resource.

## Output format

```
---
commit:            a1b2c3d
size_mb:           394.61
compression_ratio: 1.757
top1_agreement:    0.9116
kl_div:            0.0277
gen_agreement:     0.6867
gen_sanity_pass:   1.000
lambada_acc:       0.4400
tokens_per_sec:    2.6
speed_ratio:       0.213
peak_vram_mb:      2103.4
quality_gate:      FAIL
total_seconds:     379.4
gate_fail:         speed_ratio 0.213 < 0.8
```

Extract the key fields from the log file:

```
grep -E "^(compression_ratio|top1_agreement|kl_div|speed_ratio|quality_gate|gate_fail):" run.log
```

`gate_fail:` lines tell you exactly which constraint broke, and
`sanity_fail:` lines print the prompt and completion that degenerated. Read
them rather than guessing.

## Logging results

Log every experiment to `results.tsv` (tab-separated, NOT comma-separated).
Do **not** commit this file; leave it untracked.

Header, 13 columns:

```
commit	size_mb	compression_ratio	top1_agree	kl_div	gen_agree	gen_sanity	lambada	tok_per_sec	speed_ratio	quality_gate	status	description
```

Use `0` for every numeric field on a crash. `status` is `keep`, `discard`,
or `crash`. Example:

```
commit	size_mb	compression_ratio	top1_agree	kl_div	gen_agree	gen_sanity	lambada	tok_per_sec	speed_ratio	quality_gate	status	description
8d5b2f8	693.51	1.000	1.0000	0.0000	1.000	1.000	0.4200	12.8	1.000	PASS	keep	baseline
b2c3d4e	394.61	1.757	0.9116	0.0277	0.687	1.000	0.4400	2.6	0.213	FAIL	discard	bnb 8-bit weight quantization
c3d4e5f	262.37	2.643	0.8055	0.1516	0.534	0.917	0.3933	12.7	0.882	FAIL	discard	bnb 4-bit NF4, no recovery
d4e5f6g	0	0	0	0	0	0	0	0	0	FAIL	crash	structured head pruning (shape mismatch)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `compress/mar5`).

LOOP FOREVER:

1. Look at the git state: current branch/commit, and the frontier of kept experiments so far.
2. Tune `compress.py` with an experimental idea by directly hacking the code.
3. git commit.
4. Run: `uv run compress.py > run.log 2>&1` (redirect everything - do NOT use tee or let output flood your context).
5. Read the results: `grep -E "^(compression_ratio|top1_agreement|kl_div|speed_ratio|quality_gate|gate_fail|sanity_fail):" run.log`.
6. If that grep is empty, the run crashed. `tail -n 50 run.log` for the stack trace and attempt a fix. If you cannot get it working in a few attempts, give up on the idea and log it as `crash`.
7. Record the results in the tsv (do not commit results.tsv).
8. Decide keep/discard:
   - `quality_gate: FAIL` -> `discard`. The model is broken, not merely a poor trade.
   - `quality_gate: PASS` and not dominated by any kept experiment -> `keep`, advance the branch.
   - `quality_gate: PASS` but dominated on all three frontier axes -> `discard`.
9. If `discard`, `git reset --hard` back to where you started before trying the next idea.

**Timeout**: each experiment should take ~10 minutes. If a run exceeds 20
minutes, kill it and treat it as a failure. Watch for pathological logging:
bitsandbytes prints a cast warning **per quantized matmul per generated
token**, which has previously ballooned a run to 300k+ log lines and stalled
it on I/O rather than compute.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the
human whether to continue. The human might be asleep and expects you to work
*indefinitely* until manually stopped. If you run out of ideas, think
harder: attack 4-bit's repetition loop with a properly-sized distillation
budget; try mixed precision per layer (embeddings and the final layer are
more sensitive than middle MLP blocks); try structured pruning at
granularities other than whole layers (attention heads, MLP intermediate
channels); try distilling into a shallower student initialized from the
teacher's own layers. The loop runs until the human interrupts you, period.
