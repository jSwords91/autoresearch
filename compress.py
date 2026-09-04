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
    speed_ratio, fidelity,
)

t_start = time.time()

# ---------------------------------------------------------------------------
# Compression technique — EDIT THIS FUNCTION. Everything is fair game:
# quantization, pruning, low-rank factorization, distillation, architecture
# surgery, or any combination. See program.md for the rules of the loop.
# ---------------------------------------------------------------------------

def _activation_scales(model, tokenizer, n_seqs=8):
    """Mean |activation| per input channel for every Linear in the blocks.

    Quantization error only matters in proportion to what multiplies it. A
    weight column that never sees large activations can be reconstructed
    badly at no cost; one that does is expensive. This is the AWQ insight,
    used here purely as a measurement rather than to rescale anything.
    """
    from datasets import load_dataset

    scales, hooks = {}, []

    def mk_hook(name):
        def hook(mod, args):
            x = args[0].detach()
            s = x.abs().float().reshape(-1, x.shape[-1]).mean(0)
            scales[name] = scales[name] + s if name in scales else s
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "layers." in name:
            hooks.append(mod.register_forward_pre_hook(mk_hook(name)))

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="train", streaming=True)
    device = next(model.parameters()).device
    n = 0
    with torch.no_grad():
        for row in ds:
            text = row["text"].strip()
            if len(text) < 500:
                continue
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=512).input_ids.to(device)
            model(ids)
            n += 1
            if n >= n_seqs:
                break

    for h in hooks:
        h.remove()
    return scales


def _layer_sensitivity(model, scales):
    """Rank transformer blocks by how much NF4 actually hurts them.

    Uses bitsandbytes' own quantize/dequantize to get the exact NF4
    reconstruction error rather than a proxy, weighted by the measured
    activation scale, so this is the real error the network would see. Costs
    one calibration pass plus some elementwise math - no model rebuilds,
    which is what makes a 32-layer sweep affordable inside one experiment.
    """
    import bitsandbytes.functional as bnbF
    import re
    from collections import defaultdict

    per_layer = defaultdict(float)
    for name, mod in model.named_modules():
        if not (isinstance(mod, torch.nn.Linear) and name in scales):
            continue
        W = mod.weight.data
        q, state = bnbF.quantize_4bit(W.to(torch.bfloat16), quant_type="nf4")
        W_hat = bnbF.dequantize_4bit(q, state).to(torch.float32)
        err = (W.float() - W_hat).abs()
        # err is (out, in); scale is per input channel
        weighted = (err * scales[name].to(err.device).unsqueeze(0)).pow(2).sum().sqrt()
        idx = int(re.search(r"layers\.(\d+)\.", name).group(1))
        per_layer[idx] += weighted.item()

    return per_layer


N_PROTECT = 4


def compress(model, tokenizer):
    """Sensitivity-guided mixed precision: measure which transformer blocks
    NF4 actually damages, then spend bf16 on only the worst offenders and
    quantize everything else to 4 bits.

    Last session this was attempted by guessing (first/last layer, on the
    folk belief that they are most sensitive) and it barely moved the
    needle. Here the choice is measured: exact NF4 reconstruction error
    weighted by real activation magnitudes, which is the error the network
    actually experiences.
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from prepare import MODEL_CACHE_DIR

    scales = _activation_scales(model, tokenizer)
    sens = _layer_sensitivity(model, scales)

    ranked = sorted(sens.items(), key=lambda kv: kv[1], reverse=True)
    protect = [idx for idx, _ in ranked[:N_PROTECT]]
    print(f"layer sensitivity (worst first): {[(i, round(v, 1)) for i, v in ranked[:8]]}")
    print(f"protecting layers in bf16: {sorted(protect)}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # lm_head must stay out of the skip list's blast radius: passing
    # llm_int8_skip_modules REPLACES the list transformers would otherwise
    # derive automatically, which normally protects tied output embeddings.
    # Quantizing lm_head here breaks its tie to embed_tokens and the packed
    # 4-bit param fails to round-trip through save/reload.
    skip = [f"model.layers.{i}" for i in protect] + ["lm_head"]
    return AutoModelForCausalLM.from_pretrained(
        MODEL_CACHE_DIR,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=skip,
        ),
        device_map="auto",
    )


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

    # Start from a clean directory. out_dir is keyed by commit, so re-running
    # at the same commit would otherwise leave orphaned files from the
    # previous attempt (a stale extra shard, say) counting toward size_mb.
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    # Drop the in-memory model: evaluate_checkpoint reloads from disk, so the
    # thing scored is the same artifact whose bytes we are counting. Freeing
    # it here also keeps peak VRAM down, since eval loads a teacher too.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics = evaluate_checkpoint(tokenizer, out_dir)

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
    print(f"fidelity:          {fidelity(metrics):.4f}")
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
