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
    must be run with this untouched — it establishes the baseline row in
    results.tsv (compression_ratio == 1.0, quality_gate == PASS by definition).
    """
    return model


# ---------------------------------------------------------------------------
# Optional building block: a short quality-recovery fine-tune after a lossy
# compression step (e.g. after pruning or low-rank factorization). Not every
# technique needs this (plain post-training quantization usually doesn't).
# Trains on WikiText-103, which is disjoint from the WikiText-2 test split
# used for eval, so this doesn't leak into the quality gate. Not invoked by
# default — call it from compress() with whatever time budget you have left.
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

    gate_ok = passes_quality_gate(metrics, baseline)
    ratio = compression_ratio(metrics, baseline)

    elapsed = time.time() - t_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0

    print("---")
    print(f"commit:            {commit}")
    print(f"size_mb:           {metrics['size_mb']:.2f}")
    print(f"compression_ratio: {ratio:.3f}")
    print(f"bpb_wikitext2:     {metrics['bpb_wikitext2']:.6f}")
    print(f"lambada_acc:       {metrics['lambada_acc']:.4f}")
    print(f"gen_sanity_pass:   {metrics['gen_sanity_pass_rate']:.3f}")
    print(f"tokens_per_sec:    {metrics['tokens_per_sec']:.1f}")
    print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
    print(f"quality_gate:      {'PASS' if gate_ok else 'FAIL'}")
    print(f"total_seconds:     {elapsed:.1f}")

    if elapsed > TIME_BUDGET:
        print(f"WARNING: exceeded time budget of {TIME_BUDGET}s")

    if not gate_ok:
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
