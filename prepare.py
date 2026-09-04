"""
One-time setup and fixed evaluation harness for autoresearch-compress.

Downloads the baseline pretrained model + eval corpora, computes and caches
baseline metrics, and exposes the ground-truth evaluation functions used by
compress.py.

The evaluation is built around one idea: compression has a privileged
reference that general model evaluation does not - the original model. The
question is not "is the compressed model good", it is "is it the same as
this specific frozen artifact". Measuring absolute quality (perplexity,
benchmark accuracy) conflates the teacher's own errors with damage we
caused; measuring agreement with the teacher isolates the damage.

Agreement is measured at three levels of strictness, which correspond to
three distinct ways a compressed model can be wrong:

  1. top1_agreement  - does it emit the same token? (behavioural: this is
                       what a user sees under greedy decoding)
  2. kl_div          - does it hold the same distribution? (catches loss of
                       confidence/calibration that top-1 hides, and which
                       shows up under sampling)
  3. gen_agreement   - does it still say the same thing when running free?
                       (both metrics above are teacher-forced and therefore
                       blind to compounding error)

plus a degeneration check (repetition loops etc.), and two cost axes that
are reported and gated: size on disk and generation throughput. Throughput
is gated because a "compressed" model that is several times slower is not
obviously a win, and an objective that only counts bytes will happily buy
size with speed.

Usage:
    python prepare.py                  # full setup: download + baseline eval
    python prepare.py --force-baseline # recompute baseline metrics

Model, data, and baseline metrics are stored in ~/.cache/autoresearch-compress/.
"""

import os
import json
import time
import string
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"

TIME_BUDGET = 600  # compression experiment time budget in seconds (10 minutes):
                    # compress() + optional recovery + save + eval, wall clock

# --- Quality gate ----------------------------------------------------------
# The gate detects BREAKAGE. It does not measure value. Those are different
# questions and conflating them was the flaw in the previous harness: a
# single tight bar produced 17 undifferentiated FAILs across a whole session
# and no way to rank them, which gave the search nothing to climb.
#
# So: the gate answers "is this model broken?" and is deliberately loose.
# Ranking among non-broken experiments is done on the frontier - compression
# ratio against top1_agreement, kl_div and speed_ratio - which is why those
# are reported for every run whether it passes or not.
#
# Thresholds are calibrated from measured behaviour, not chosen as round
# numbers. Reference points on SmolLM2-360M-Instruct:
#
#   technique   compress  top1    kl      gen_agr  sanity  speed
#   8-bit bnb   1.757x    0.9116  0.0277  0.6867   1.000   0.213
#   4-bit NF4   2.643x    0.8055  0.1516  0.5339   0.917   0.882
#
# 4-bit emits a visible repetition loop, so real breakage sits somewhere
# between KL 0.03 and 0.15 for this model. gen_sanity catches that case
# directly, so KL_CEILING is set well above it as a backstop for
# catastrophic distribution damage (the structural-surgery failures from the
# previous harness were an order of magnitude worse than either row above).
#
# top1_agreement and gen_agreement are NOT gated. Free-running agreement in
# particular is inherently low even for good compressions, because greedy
# decoding is chaotic: one different token cascades through everything after
# it. They are frontier axes, not pass/fail criteria.

KL_CEILING = 0.25               # backstop for catastrophic distribution damage
GEN_SANITY_TOLERANCE_ABS = 0.0  # degeneration rate must not regress at all
SPEED_FLOOR_RATIO = 0.60        # reject techniques that buy size with speed.
                                # Set to catch the pathological case (bnb
                                # int8's per-matmul dequant lands at 0.213)
                                # without blocking 4-bit work, which sits
                                # around 0.75-0.88. Measurement is +/-2.5%
                                # after interleaving, so this is ~15 sigma
                                # from where real techniques live.

# --- Eval set sizes --------------------------------------------------------
# The fidelity corpus is deliberately mixed: chat-formatted conversations
# (the model is instruction-tuned, so that is the distribution that matters)
# and general prose (catches damage that only shows up on long-form text).
# Every sequence contributes hundreds of token positions, so these small
# counts still yield tens of thousands of measurements - far denser than a
# few hundred binary accuracy outcomes.
FIDELITY_CHAT_SEQS = 40
FIDELITY_PROSE_SEQS = 40
FIDELITY_MAX_LEN = 512

GEN_MAX_NEW_TOKENS = 128     # cap on generated tokens for the sanity suite
GEN_AGREEMENT_TOKENS = 64    # tokens to free-run when comparing against teacher
LAMBADA_EVAL_EXAMPLES = 150  # reported for legibility, NOT gated (see below)

_REPEAT_NGRAM_N = 4            # n-gram size used by the degenerate-output detector
_REPEAT_NGRAM_MAX_RATIO = 0.5  # flag as degenerate if >50% of n-grams repeat

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-compress")
MODEL_CACHE_DIR = os.path.join(CACHE_DIR, "baseline-model")
BASELINE_METRICS_PATH = os.path.join(CACHE_DIR, "baseline.json")
CHECKPOINTS_DIR = os.path.join(CACHE_DIR, "checkpoints")

DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Fixed instruction-prompt suite. Deliberately varied (facts, formatting,
# arithmetic, translation) so a technique that breaks one narrow capability
# still gets caught. Used for both the degeneration check and the
# free-running agreement check.
SANITY_PROMPTS = [
    "What is the capital of France?",
    "Write a haiku about the ocean.",
    "List three primary colors.",
    "Reverse the word 'hello'.",
    "What is 12 plus 7?",
    "Name one planet in the solar system.",
    "Complete the sentence: The sky is",
    "Give a one-sentence summary of what a computer is.",
    "What is the opposite of 'hot'?",
    "Translate 'good morning' to French.",
    "Count from 1 to 5.",
    "What day comes after Monday?",
]

# ---------------------------------------------------------------------------
# Model / tokenizer loading
# ---------------------------------------------------------------------------

def download_baseline():
    """One-time download of the baseline model + tokenizer into the cache dir."""
    if os.path.exists(os.path.join(MODEL_CACHE_DIR, "config.json")):
        print(f"Model: already cached at {MODEL_CACHE_DIR}")
        return
    print(f"Model: downloading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    tokenizer.save_pretrained(MODEL_CACHE_DIR)
    model.save_pretrained(MODEL_CACHE_DIR)
    print(f"Model: saved to {MODEL_CACHE_DIR}")


def load_baseline():
    """Load the cached baseline model + tokenizer onto the best available device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_CACHE_DIR, torch_dtype=DTYPE)
    model.to(device)
    model.eval()
    return model, tokenizer


def load_teacher():
    """Load a second, frozen copy of the baseline to compare against. This is
    the reference the whole evaluation is defined relative to."""
    model, _ = load_baseline()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_checkpoint(model_dir):
    """Load a saved checkpoint back off disk, the way a downstream user
    would. `torch_dtype="auto"` honours the dtype recorded in config.json;
    quantized checkpoints carry a quantization_config that dictates their own
    dtype and placement, so they are loaded through device_map instead."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)

    if "quantization_config" in cfg:
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto")
        model.to(device)
    model.eval()
    return model


_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def _dir_files(d):
    """Relative path -> size for every file under a directory."""
    out = {}
    for root, _, files in os.walk(d):
        for f in files:
            path = os.path.join(root, f)
            out[os.path.relpath(path, d)] = os.path.getsize(path)
    return out


def assert_checkpoint_parity(model_dir):
    """Refuse to score a checkpoint whose file set is not comparable to the
    baseline's.

    size_mb sums every file in a directory, so a checkpoint that omits the
    tokenizer is credited several free MB against a baseline directory that
    includes it (tokenizer.json alone is 3.4MB here, about 1.3% of a 4-bit
    checkpoint). Worse, a directory containing no weights at all would score
    a near-infinite compression ratio: an earlier crashed run left behind a
    directory holding nothing but config.json and generation_config.json,
    which would have been measured as a ~680,000x compression.

    Neither failure is a property of the compressed model, so both raise
    rather than failing the gate - the experiment is invalid, not merely bad.
    """
    baseline_files = _dir_files(MODEL_CACHE_DIR)
    ckpt_files = _dir_files(model_dir)

    required = {f for f in baseline_files if not f.endswith(_WEIGHT_SUFFIXES)}
    missing = sorted(required - set(ckpt_files))
    if missing:
        raise RuntimeError(
            f"checkpoint {model_dir} is missing files present in the baseline: "
            f"{missing}. size_mb would not be comparable. Save the tokenizer "
            f"and config alongside the weights."
        )

    if not [f for f in ckpt_files if f.endswith(_WEIGHT_SUFFIXES)]:
        raise RuntimeError(
            f"checkpoint {model_dir} contains no weight files "
            f"({'/'.join(_WEIGHT_SUFFIXES)}); there is no model here to score."
        )


def _model_device(model):
    """Infer the model's device rather than assuming one, so eval works no
    matter what device a compression technique leaves the model on (e.g.
    CPU-only dynamic quantization)."""
    return next(model.parameters()).device

# ---------------------------------------------------------------------------
# Eval data
# ---------------------------------------------------------------------------

def _fidelity_corpus(tokenizer):
    """Mixed chat + prose corpus, tokenized. Returns a list of 1 x T id tensors.

    Chat sequences use the model's own chat template because the baseline is
    instruction-tuned: that is the distribution we actually care about
    preserving. Prose sequences catch damage that only shows on long-form
    text. Uses held-out splits so a recovery fine-tune training on the
    corresponding train splits does not leak into the gate.
    """
    seqs = []

    chat = load_dataset("HuggingFaceH4/no_robots", split="test", streaming=True)
    for row in chat:
        encoded = tokenizer.apply_chat_template(
            row["messages"], return_tensors="pt", truncation=True,
            max_length=FIDELITY_MAX_LEN, return_dict=True,
        )
        ids = encoded["input_ids"]
        if ids.size(1) >= 32:
            seqs.append(ids)
        if len(seqs) >= FIDELITY_CHAT_SEQS:
            break

    n_chat = len(seqs)
    prose = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    for row in prose:
        text = row["text"].strip()
        if len(text) < 500:
            continue
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=FIDELITY_MAX_LEN).input_ids
        if ids.size(1) >= 32:
            seqs.append(ids)
        if len(seqs) - n_chat >= FIDELITY_PROSE_SEQS:
            break

    return seqs


def _lambada_examples():
    ds = load_dataset("EleutherAI/lambada_openai", "default", split="test")
    return [row["text"] for row in ds][:LAMBADA_EVAL_EXAMPLES]

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE - these are the ground-truth metrics)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_fidelity(model, teacher, corpus):
    """
    Teacher-forced agreement between the compressed model and the frozen
    baseline, at two levels of strictness.

    top1_agreement: fraction of positions where both models' argmax matches.
    Directly interpretable and behaviourally meaningful - it is exactly what
    determines greedy-decode output.

    kl_div: mean forward KL(teacher || student) in nats per token. Forward
    (mode-covering) is the correct direction for compression: we want the
    student to reproduce everything the teacher would say, not merely find
    one mode of it. Catches loss of confidence that top-1 agreement is blind
    to.

    Every sequence contributes hundreds of positions, so this is a far denser
    measurement than any accuracy metric at comparable cost.
    """
    device = _model_device(model)
    t_device = _model_device(teacher)

    n_positions = 0
    n_agree = 0
    kl_total = 0.0

    for ids in corpus:
        s_logits = model(ids.to(device)).logits.float()
        t_logits = teacher(ids.to(t_device)).logits.float().to(s_logits.device)

        # drop the final position: it predicts a token we do not have
        s_logits = s_logits[0, :-1, :]
        t_logits = t_logits[0, :-1, :]
        if s_logits.size(0) == 0:
            continue

        n_agree += (s_logits.argmax(-1) == t_logits.argmax(-1)).sum().item()

        t_logprob = F.log_softmax(t_logits, dim=-1)
        s_logprob = F.log_softmax(s_logits, dim=-1)
        kl = (t_logprob.exp() * (t_logprob - s_logprob)).sum(-1)
        kl_total += kl.sum().item()

        n_positions += s_logits.size(0)

    return {
        "top1_agreement": n_agree / n_positions if n_positions else 0.0,
        "kl_div": kl_total / n_positions if n_positions else float("inf"),
        "fidelity_positions": n_positions,
    }


@torch.no_grad()
def compute_generation_agreement(model, teacher, tokenizer, prompts=None):
    """
    Free-running agreement: generate greedily from both models on the same
    prompts and measure how long they stay identical.

    Teacher-forced metrics cannot see compounding error - a model can match
    the teacher at 99% of positions when fed ground truth, then diverge into
    garbage once it starts consuming its own output. This is the metric that
    catches that. Score is the mean normalised prefix length before
    divergence (1.0 = identical generations).
    """
    device = _model_device(model)
    t_device = _model_device(teacher)
    prompts = prompts if prompts is not None else SANITY_PROMPTS

    scores = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        ids = encoded["input_ids"]
        n_prompt = ids.size(1)

        try:
            s_gen = model.generate(
                ids.to(device), max_new_tokens=GEN_AGREEMENT_TOKENS,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )[0, n_prompt:].tolist()
            t_gen = teacher.generate(
                ids.to(t_device), max_new_tokens=GEN_AGREEMENT_TOKENS,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )[0, n_prompt:].tolist()
        except RuntimeError:
            scores.append(0.0)
            continue

        match = 0
        for s_tok, t_tok in zip(s_gen, t_gen):
            if s_tok != t_tok:
                break
            match += 1
        scores.append(match / max(len(t_gen), 1))

    return sum(scores) / len(scores) if scores else 0.0


def _is_degenerate(text, n=_REPEAT_NGRAM_N, max_ratio=_REPEAT_NGRAM_MAX_RATIO):
    words = text.split()
    if len(words) < n + 1:
        return False
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    unique_ratio = len(set(ngrams)) / len(ngrams)
    return unique_ratio < (1 - max_ratio)


@torch.no_grad()
def run_generation_sanity(model, tokenizer, prompts=None):
    """
    Absolute (not comparative) check that output has not collapsed: non-empty,
    non-degenerate, and generate() did not raise. Catches repetition loops,
    which are the classic way a quantized model breaks.
    """
    device = _model_device(model)
    prompts = prompts if prompts is not None else SANITY_PROMPTS
    results = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        input_ids = encoded["input_ids"].to(device)
        try:
            gen = model.generate(
                input_ids,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        except RuntimeError as e:
            results.append({"prompt": prompt, "ok": False, "reason": f"generate() raised: {e}"})
            continue
        completion = tokenizer.decode(gen[0, input_ids.size(1):], skip_special_tokens=True)
        ok = bool(completion.strip()) and not _is_degenerate(completion)
        results.append({"prompt": prompt, "ok": ok, "completion": completion[:200]})
    pass_rate = sum(r["ok"] for r in results) / len(results)
    return {"pass_rate": pass_rate, "details": results}


@torch.no_grad()
def compute_lambada_accuracy(model, tokenizer, examples=None):
    """
    LAMBADA last-word cloze accuracy. Reported for human legibility (it is a
    number people have intuitions about) but deliberately NOT part of the
    quality gate: a few hundred binary outcomes cannot resolve the
    differences we care about, and agreement-with-teacher already covers
    capability preservation far more densely.
    """
    device = _model_device(model)
    examples = examples if examples is not None else _lambada_examples()
    correct = 0
    total = 0
    for text in examples:
        text = text.strip()
        if " " not in text:
            continue
        context, target = text.rsplit(" ", 1)
        target_ids = tokenizer(" " + target, add_special_tokens=False).input_ids
        if not target_ids:
            continue
        input_ids = tokenizer(context, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(device)
        gen = model.generate(
            input_ids,
            max_new_tokens=len(target_ids),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        completion = tokenizer.decode(gen[0, input_ids.size(1):], skip_special_tokens=True)
        predicted_words = completion.strip().split()
        predicted_word = predicted_words[0].strip(string.punctuation) if predicted_words else ""
        target_word = target.strip(string.punctuation)
        if predicted_word == target_word:
            correct += 1
        total += 1
    return correct / total if total else 0.0


def measure_size_mb(model_dir):
    """On-disk size of a saved checkpoint directory (weights + config), in MB.
    Deliberately size-on-disk rather than parameter count: it is the one
    metric that is honest across quantization, structured pruning, and
    distillation alike. Note that UNSTRUCTURED pruning alone does not shrink
    this number unless the result is serialized in a genuinely sparse format
    - see program.md."""
    total_bytes = 0
    for root, _, files in os.walk(model_dir):
        for f in files:
            total_bytes += os.path.getsize(os.path.join(root, f))
    return total_bytes / (1024 * 1024)


@torch.no_grad()
def _time_generation(model, tokenizer, input_ids, max_new_tokens):
    device = _model_device(model)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    gen = model.generate(input_ids.to(device), max_new_tokens=max_new_tokens,
                         do_sample=False, pad_token_id=tokenizer.eos_token_id)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    return (gen.size(1) - input_ids.size(1)) / dt


@torch.no_grad()
def measure_relative_throughput(model, reference, tokenizer,
                                prompt="The quick brown fox jumps over the lazy dog. ",
                                max_new_tokens=100, repeats=3):
    """Throughput of `model` and `reference` on the same prompt, INTERLEAVED
    and reduced by median.

    Measuring one then the other gave a ratio of 0.882 and 0.753 on identical
    code, because this GPU throttles continuously (2100MHz -> 210MHz over a
    session): whichever model is measured second is systematically penalised,
    and the bias varies with how hot the card already is. Interleaving
    cancels monotonic drift, and the median rejects one-off stalls. Returns
    (model_tps, reference_tps).
    """
    ids = tokenizer(prompt, return_tensors="pt").input_ids

    # warm both, so neither pays first-call compilation/allocation costs
    for m in (reference, model):
        _time_generation(m, tokenizer, ids, max_new_tokens)

    ref_runs, mdl_runs = [], []
    for _ in range(repeats):
        ref_runs.append(_time_generation(reference, tokenizer, ids, max_new_tokens))
        mdl_runs.append(_time_generation(model, tokenizer, ids, max_new_tokens))

    ref_runs.sort()
    mdl_runs.sort()
    mid = len(ref_runs) // 2
    return mdl_runs[mid], ref_runs[mid]


def evaluate_checkpoint(tokenizer, model_dir, corpus=None):
    """
    Single entry point compress.py calls. Returns every metric needed to
    apply the quality gate and log results.tsv. DO NOT CHANGE the
    definitions of the individual metric functions above.

    The model is RELOADED FROM model_dir and that reloaded model is what
    gets scored. Scoring an in-memory object while separately measuring a
    directory's size would let the quality numbers and the size number refer
    to different artifacts - and size on disk is the objective, so the two
    halves of "quality per byte" must describe the same thing. Reloading
    also means a technique that produces an unloadable checkpoint fails
    loudly here rather than being scored on a model that only ever existed
    in memory.

    Ordering matters and is deliberate: everything needing the teacher runs
    first, then the teacher is freed, and only then is throughput measured.
    Otherwise the teacher's ~700MB sits on the GPU during the latency test
    and depresses it, by an amount that varies with how big the compressed
    model is - which would make a smaller model look faster purely because
    it left more room. Since throughput is gated, that confound would
    produce wrong verdicts.

    `corpus` is rebuilt here if not supplied; pass it in to avoid
    re-streaming the datasets.
    """
    assert_checkpoint_parity(model_dir)
    model = load_checkpoint(model_dir)

    if corpus is None:
        corpus = _fidelity_corpus(tokenizer)

    teacher = load_teacher()
    fidelity = compute_fidelity(model, teacher, corpus)
    gen_agreement = compute_generation_agreement(model, teacher, tokenizer)

    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # teacher-free from here: these measure the compressed model alone
    sanity = run_generation_sanity(model, tokenizer)
    lambada_acc = compute_lambada_accuracy(model, tokenizer)
    size_mb = measure_size_mb(model_dir)

    # Throughput is measured against a reference copy loaded RIGHT NOW,
    # rather than against the figure cached in baseline.json. On a thermally
    # throttled laptop GPU the absolute rate drifts enormously across a
    # session (measured: 30 tok/s cold, 12 tok/s after hours of runs, as the
    # SM clock drops from 2100MHz to 210MHz). A ratio only cancels that
    # common-mode drift if both halves are measured in the same thermal
    # state, seconds apart. Both are measured with both models resident so
    # the memory footprint is symmetric too.
    reference = load_teacher()
    tok_per_sec, ref_tok_per_sec = measure_relative_throughput(model, reference, tokenizer)
    del reference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "reference_tokens_per_sec": ref_tok_per_sec,
        "top1_agreement": fidelity["top1_agreement"],
        "kl_div": fidelity["kl_div"],
        "fidelity_positions": fidelity["fidelity_positions"],
        "gen_agreement": gen_agreement,
        "gen_sanity_pass_rate": sanity["pass_rate"],
        "gen_sanity_details": sanity["details"],
        "lambada_acc": lambada_acc,
        "size_mb": size_mb,
        "tokens_per_sec": tok_per_sec,
    }


def passes_quality_gate(metrics, baseline):
    """Breakage detector. Returns (ok, reasons) so callers can report WHY
    something failed rather than only that it did.

    Passing means "not broken", NOT "good". Ranking among passing
    experiments is done on the frontier - see rank_key()."""
    reasons = []

    if metrics["kl_div"] > KL_CEILING:
        reasons.append(f"kl_div {metrics['kl_div']:.4f} > {KL_CEILING}")
    if metrics["gen_sanity_pass_rate"] < baseline["gen_sanity_pass_rate"] - GEN_SANITY_TOLERANCE_ABS:
        reasons.append(
            f"gen_sanity {metrics['gen_sanity_pass_rate']:.3f} < "
            f"{baseline['gen_sanity_pass_rate'] - GEN_SANITY_TOLERANCE_ABS:.3f}"
        )
    ratio = speed_ratio(metrics, baseline)
    if ratio < SPEED_FLOOR_RATIO:
        reasons.append(f"speed_ratio {ratio:.3f} < {SPEED_FLOOR_RATIO}")

    return len(reasons) == 0, reasons


def compression_ratio(metrics, baseline):
    """The objective to maximize, once the quality gate passes."""
    return baseline["size_mb"] / metrics["size_mb"]


def frontier_point(metrics, baseline):
    """The three axes an experiment is ranked on, all higher-is-better.

    Compression is what we are buying; fidelity and speed are what we pay
    with. Collapsing these into one scalar would bake in an exchange rate
    nobody has justified, so they stay separate and ranking is by Pareto
    dominance instead.
    """
    return {
        "compression_ratio": compression_ratio(metrics, baseline),
        "top1_agreement": metrics["top1_agreement"],
        "speed_ratio": speed_ratio(metrics, baseline),
    }


def dominates(a, b):
    """True if frontier point `a` is at least as good as `b` on every axis
    and strictly better on at least one."""
    keys = ("compression_ratio", "top1_agreement", "speed_ratio")
    return all(a[k] >= b[k] for k in keys) and any(a[k] > b[k] for k in keys)


def speed_ratio(metrics, baseline=None):
    """Throughput relative to the uncompressed model. Above 1.0 means the
    compressed model is also faster, which is the ideal outcome.

    Uses the reference measured during the same run when available, since
    the cached baseline figure was taken in a different thermal state and is
    not comparable (see evaluate_checkpoint). `baseline` is accepted only as
    a fallback for metrics dicts predating that change.
    """
    ref = metrics.get("reference_tokens_per_sec")
    if not ref:
        ref = baseline["tokens_per_sec"]
    return metrics["tokens_per_sec"] / ref

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare baseline model, data, and baseline metrics")
    parser.add_argument("--force-baseline", action="store_true", help="Recompute baseline metrics even if cached")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()

    download_baseline()
    print()

    if os.path.exists(BASELINE_METRICS_PATH) and not args.force_baseline:
        print(f"Baseline: already computed at {BASELINE_METRICS_PATH}")
    else:
        print("Baseline: evaluating the unmodified model against itself.")
        print("Agreement must come out at exactly 1.0 and KL at 0.0 by definition,")
        print("so this doubles as a self-test that the harness is wired up correctly.")
        _, tokenizer = load_baseline()
        corpus = _fidelity_corpus(tokenizer)
        metrics = evaluate_checkpoint(tokenizer, MODEL_CACHE_DIR, corpus=corpus)
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(BASELINE_METRICS_PATH, "w") as f:
            json.dump({k: v for k, v in metrics.items() if k != "gen_sanity_details"}, f, indent=2)
        print(f"Baseline: saved to {BASELINE_METRICS_PATH}")
        print()
        print("---")
        for k, v in metrics.items():
            if k != "gen_sanity_details":
                print(f"{k}: {v}")

    print()
    print("Done! Ready to compress. See program.md.")
