"""
Compression experiment script. Loads the baseline model, applies a
compression technique, optionally recovers quality with a short fine-tune,
saves the result, and evaluates it against the fixed harness in prepare.py.

Usage: uv run compress.py
"""

import os
import json
import time
import shutil
import subprocess

import torch

from prepare import (
    CHECKPOINTS_DIR, BASELINE_METRICS_PATH, TIME_BUDGET,
    load_baseline, evaluate_checkpoint, passes_quality_gate, compression_ratio,
    speed_ratio,
)

t_start = time.time()

# ---------------------------------------------------------------------------
# Compression technique — EDIT THIS FUNCTION. Everything is fair game:
# quantization, pruning, low-rank factorization, distillation, architecture
# surgery, or any combination. See program.md for the rules of the loop.
# ---------------------------------------------------------------------------

def compress(model, tokenizer):
    """
    Baseline: no-op. Returns the model unchanged. The very first experiment
    must be run with this untouched - it establishes the baseline row in
    results.tsv (compression_ratio 1.0, perfect agreement by definition).
    """
    return model


# ---------------------------------------------------------------------------
# Optional building block: a short quality-recovery fine-tune after a lossy
# compression step (e.g. after pruning or low-rank factorization). Not every
# technique needs this (plain post-training quantization usually doesn't).
# Trains on WikiText-103, which is disjoint from the eval corpora, so this
# doesn't leak into the quality gate. Not invoked by default.
#
# MIND THE TOKEN BUDGET. This loop is batch-size 1, so a 4-minute run is on
# the order of 100k tokens. SmolLM2 was pretrained on ~4 trillion. Any
# structural damage big enough to need repair is very unlikely to be
# repairable at 1e-8 of the original budget, and an earlier session burned
# five experiments rediscovering that. If a technique needs recovery to
# pass, either make the damage small enough that recovery is a nudge rather
# than a rebuild, or raise the batch size / sequence packing so the token
# count is at least in the millions. Distilling against the teacher's
# distribution (KL) is a strictly richer signal per token than LM loss on
# raw text, so prefer it when you do spend the budget.
# ---------------------------------------------------------------------------

def recover_with_finetune(model, tokenizer, seconds, lr=1e-5):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    device = next(model.parameters()).device
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.time()
    for row in ds:
        text = row["text"].strip()
        if len(text) < 200:
            continue
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
        if ids.size(1) < 2:
            continue
        out = model(ids, labels=ids)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        if time.time() - t0 > seconds:
            break
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Example technique (commented out — a starting point, not a default).
# Illustrates the pattern; not guaranteed to round-trip through
# save_pretrained() as-is — quantized/sparse formats often need their own
# serialization path instead of plain safetensors. That's part of the
# experiment.
# ---------------------------------------------------------------------------

# def compress(model, tokenizer):
#     """Dynamic int8 quantization of all Linear layers (CPU-only in vanilla
#     PyTorch — bitsandbytes is available in pyproject.toml for a CUDA-capable
#     alternative)."""
#     model = model.to("cpu")
#     model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
#     return model


# ---------------------------------------------------------------------------
# Experiment runner — no need to edit below this line
# ---------------------------------------------------------------------------

def main():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model, tokenizer = load_baseline()

    model = compress(model, tokenizer)

    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    out_dir = os.path.join(CHECKPOINTS_DIR, commit)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    metrics = evaluate_checkpoint(model, tokenizer, out_dir)

    with open(BASELINE_METRICS_PATH) as f:
        baseline = json.load(f)

    gate_ok, reasons = passes_quality_gate(metrics, baseline)
    ratio = compression_ratio(metrics, baseline)
    spd = speed_ratio(metrics, baseline)

    elapsed = time.time() - t_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0

    print("---")
    print(f"commit:            {commit}")
    print(f"size_mb:           {metrics['size_mb']:.2f}")
    print(f"compression_ratio: {ratio:.3f}")
    print(f"top1_agreement:    {metrics['top1_agreement']:.4f}")
    print(f"kl_div:            {metrics['kl_div']:.4f}")
    print(f"gen_agreement:     {metrics['gen_agreement']:.4f}")
    print(f"gen_sanity_pass:   {metrics['gen_sanity_pass_rate']:.3f}")
    print(f"lambada_acc:       {metrics['lambada_acc']:.4f}")
    print(f"tokens_per_sec:    {metrics['tokens_per_sec']:.1f}")
    print(f"speed_ratio:       {spd:.3f}")
    print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
    print(f"quality_gate:      {'PASS' if gate_ok else 'FAIL'}")
    print(f"total_seconds:     {elapsed:.1f}")

    # Report WHY the gate failed, and which sanity prompts broke. Without
    # this you cannot tell a real regression from a threshold set too tight.
    for r in reasons:
        print(f"gate_fail:         {r}")
    for d in metrics["gen_sanity_details"]:
        if not d["ok"]:
            detail = d.get("completion", d.get("reason", ""))
            print(f"sanity_fail:       {d['prompt']!r} -> {detail!r}")

    if elapsed > TIME_BUDGET:
        print(f"WARNING: exceeded time budget of {TIME_BUDGET}s")

    if not gate_ok:
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
