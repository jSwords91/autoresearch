"""
One-time setup and fixed evaluation harness for autoresearch-compress.

Downloads the baseline pretrained model + eval corpora, computes and caches
baseline metrics, and exposes the ground-truth evaluation functions used by
compress.py: bits-per-byte on held-out WikiText-2, LAMBADA cloze accuracy,
a generation-sanity check, on-disk size, and generation throughput.

Usage:
    python prepare.py                  # full setup: download + baseline eval
    python prepare.py --force-baseline # recompute baseline metrics

Model, data, and baseline metrics are stored in ~/.cache/autoresearch-compress/.
"""

import os
import math
import json
import time
import string
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

TIME_BUDGET = 600  # compression experiment time budget in seconds (10 minutes):
                    # compress() + optional recovery fine-tune + save + eval, wall clock

# Quality gate tolerances. An experiment fails the gate if ANY of these are
# violated. See program.md for the full rationale.
BPB_TOLERANCE = 0.01           # max allowed relative regression in wikitext2 bpb (1%)
LAMBADA_TOLERANCE_ABS = 0.02   # max allowed absolute drop in lambada accuracy (2 points)
GEN_SANITY_TOLERANCE_ABS = 0.0 # generation sanity pass rate must not regress at all vs baseline

# Eval set sizes. Tunable down for faster iteration on smaller GPUs, tunable
# up for a more reliable gate if you have time budget to spare.
WIKITEXT_EVAL_DOCS = 200     # held-out wikitext-2 test docs used for bpb eval
LAMBADA_EVAL_EXAMPLES = 300  # lambada test examples used for accuracy eval
GEN_MAX_NEW_TOKENS = 128     # cap on generated tokens for the sanity-check suite

_REPEAT_NGRAM_N = 4          # n-gram size used by the degenerate-output detector
_REPEAT_NGRAM_MAX_RATIO = 0.5  # flag as degenerate if >50% of n-grams repeat

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-compress")
MODEL_CACHE_DIR = os.path.join(CACHE_DIR, "baseline-model")
BASELINE_METRICS_PATH = os.path.join(CACHE_DIR, "baseline.json")
CHECKPOINTS_DIR = os.path.join(CACHE_DIR, "checkpoints")

DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Fixed instruction-prompt suite for the generation sanity check. Deliberately
# varied (facts, formatting, arithmetic, translation) so a technique that
# breaks one narrow capability still gets caught.
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


def _model_device(model):
    """Infer the model's device rather than assuming one, so eval works no
    matter what device a compression technique leaves the model on (e.g.
    CPU-only dynamic quantization)."""
    return next(model.parameters()).device

# ---------------------------------------------------------------------------
# Eval data
# ---------------------------------------------------------------------------

def _wikitext_docs():
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    docs = [row["text"] for row in ds if len(row["text"].strip()) > 200]
    return docs[:WIKITEXT_EVAL_DOCS]


def _lambada_examples():
    ds = load_dataset("EleutherAI/lambada_openai", "default", split="test")
    return [row["text"] for row in ds][:LAMBADA_EVAL_EXAMPLES]

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — these are the ground-truth metrics)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_bpb(model, tokenizer, docs=None):
    """
    Bits per byte on held-out WikiText-2 test documents: vocab-independent,
    so it stays comparable even if a technique changes tokenization or
    architecture. Approximates target bytes with the whole document's UTF-8
    byte length (a fixed, consistent denominator across baseline and every
    compressed checkpoint), which is exact enough for relative comparison.
    """
    device = _model_device(model)
    docs = docs if docs is not None else _wikitext_docs()
    total_nats = 0.0
    total_bytes = 0
    for text in docs:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(device)
        if ids.size(1) < 2:
            continue
        out = model(ids, labels=ids)
        num_targets = ids.size(1) - 1
        total_nats += out.loss.item() * num_targets
        total_bytes += len(text.encode("utf-8"))
    return total_nats / (math.log(2) * total_bytes)


@torch.no_grad()
def compute_lambada_accuracy(model, tokenizer, examples=None):
    """
    LAMBADA last-word cloze accuracy: split off the final word of each
    example, greedily generate as many tokens as the target word takes, and
    check for an exact text match (punctuation-stripped).
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
    Fixed instruction-prompt suite. A prompt 'passes' if the model produces a
    non-empty, non-degenerate, NaN-free response. This catches breakage that
    aggregate bpb/accuracy numbers can miss (e.g. coherence collapse on
    specific heads after pruning, without moving the loss much).
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
        has_nan = torch.isnan(gen.float()).any().item() if torch.is_floating_point(gen) else False
        ok = bool(completion.strip()) and not _is_degenerate(completion) and not has_nan
        results.append({"prompt": prompt, "ok": ok, "completion": completion[:200]})
    pass_rate = sum(r["ok"] for r in results) / len(results)
    return {"pass_rate": pass_rate, "details": results}


def measure_size_mb(model_dir):
    """On-disk size of a saved checkpoint directory (weights + config), in MB.
    Deliberately size-on-disk rather than parameter count: it's the one
    metric that's honest across quantization, structured pruning, and
    distillation alike. Note that UNSTRUCTURED pruning alone does not shrink
    this number unless the result is serialized in a genuinely sparse format
    — see program.md."""
    total_bytes = 0
    for root, _, files in os.walk(model_dir):
        for f in files:
            total_bytes += os.path.getsize(os.path.join(root, f))
    return total_bytes / (1024 * 1024)


@torch.no_grad()
def measure_latency(model, tokenizer, prompt="The quick brown fox jumps over the lazy dog. ", max_new_tokens=100, warmup=1):
    """Tokens/sec for greedy generation on a fixed prompt. Logged for
    interest, not part of the quality gate or the compression objective."""
    device = _model_device(model)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    for _ in range(warmup):
        model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    gen = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    n_generated = gen.size(1) - input_ids.size(1)
    return n_generated / dt


def evaluate_checkpoint(model, tokenizer, model_dir):
    """
    Single entry point compress.py calls. Returns every metric needed to
    apply the quality gate and log results.tsv. DO NOT CHANGE the
    definitions of the individual metric functions above.
    """
    bpb = compute_bpb(model, tokenizer)
    lambada_acc = compute_lambada_accuracy(model, tokenizer)
    sanity = run_generation_sanity(model, tokenizer)
    size_mb = measure_size_mb(model_dir)
    tok_per_sec = measure_latency(model, tokenizer)
    return {
        "bpb_wikitext2": bpb,
        "lambada_acc": lambada_acc,
        "gen_sanity_pass_rate": sanity["pass_rate"],
        "gen_sanity_details": sanity["details"],
        "size_mb": size_mb,
        "tokens_per_sec": tok_per_sec,
    }


def passes_quality_gate(metrics, baseline):
    """The one hard constraint. See program.md for full rationale."""
    bpb_ok = metrics["bpb_wikitext2"] <= baseline["bpb_wikitext2"] * (1 + BPB_TOLERANCE)
    lambada_ok = metrics["lambada_acc"] >= baseline["lambada_acc"] - LAMBADA_TOLERANCE_ABS
    sanity_ok = metrics["gen_sanity_pass_rate"] >= baseline["gen_sanity_pass_rate"] - GEN_SANITY_TOLERANCE_ABS
    return bpb_ok and lambada_ok and sanity_ok


def compression_ratio(metrics, baseline):
    """The objective to maximize, once the quality gate passes."""
    return baseline["size_mb"] / metrics["size_mb"]

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

    print("Data: caching WikiText-2 (test) and LAMBADA (test) via `datasets`...")
    _wikitext_docs()
    _lambada_examples()
    print("Data: ready")
    print()

    if os.path.exists(BASELINE_METRICS_PATH) and not args.force_baseline:
        print(f"Baseline: already computed at {BASELINE_METRICS_PATH}")
    else:
        print("Baseline: evaluating unmodified model (this establishes the results.tsv baseline row)...")
        model, tokenizer = load_baseline()
        metrics = evaluate_checkpoint(model, tokenizer, MODEL_CACHE_DIR)
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(BASELINE_METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Baseline: saved to {BASELINE_METRICS_PATH}")
        print()
        print("---")
        for k, v in metrics.items():
            if k != "gen_sanity_details":
                print(f"{k}: {v}")

    print()
    print("Done! Ready to compress. See program.md.")
