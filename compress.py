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

# --- Activation-aware scaling (AWQ-style) ----------------------------------
# Rather than spending bf16 to protect sensitive weights, make 4 bits go
# further. NF4 quantizes in blocks with a shared absmax scale, so within a
# block the channels with the largest magnitude get the most of the grid.
# If we scale weight column j up by s_j and fold 1/s_j into whatever produced
# that input channel, the network computes exactly the same function, but the
# quantizer now spends its resolution on the channels that actually carry
# signal.
#
# Two foldings are exact here:
#   q/k/v   <- input_layernorm.weight          (RMSNorm is elementwise)
#   gate/up <- post_attention_layernorm.weight
#   down    <- up_proj output rows   (down's input is act(gate(x)) * up(x),
#                                     so scaling up's row j by 1/s scales
#                                     that product channel by 1/s)
#
# down_proj is the point of this: it was the single worst projection in the
# sensitivity ranking, and unlike the others it has no norm in front of it,
# so it is the one plain AWQ implementations usually skip.

ALPHA = 0.25
SCALE_DOWN_PROJ = False
DOUBLE_QUANT = True
PROTECT_WORST = ["model.layers.31.mlp.down_proj"]


def _norm_input_acts(model, tokenizer, n_seqs=8):
    """Mean |activation| per channel at each point we can fold a scale into."""
    acts, hooks = {}, []

    def mk(name):
        def hook(mod, args):
            x = args[0].detach()
            v = x.abs().float().reshape(-1, x.shape[-1]).mean(0)
            acts[name] = acts[name] + v if name in acts else v
        return hook

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.q_proj.register_forward_pre_hook(mk(f"{i}.attn_in")))
        hooks.append(layer.mlp.gate_proj.register_forward_pre_hook(mk(f"{i}.mlp_in")))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(mk(f"{i}.down_in")))

    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="train", streaming=True)
    device = next(model.parameters()).device
    n = 0
    with torch.no_grad():
        for row in ds:
            if len(row["text"].strip()) < 500:
                continue
            ids = tokenizer(row["text"].strip(), return_tensors="pt",
                            truncation=True, max_length=512).input_ids.to(device)
            model(ids)
            n += 1
            if n >= n_seqs:
                break
    for h in hooks:
        h.remove()
    return acts


def _scale_from(act, alpha):
    """s = act^alpha, normalised to geometric mean 1 so the folding neither
    inflates nor deflates the tensors overall."""
    a = act.float().clamp(min=1e-5)
    s = a.pow(alpha)
    s = s / s.log().mean().exp()
    return s.clamp(min=1e-3, max=1e3)


ALPHA_GRID = (0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7)


def _best_alpha(weights, act, grid=ALPHA_GRID):
    """Pick the scaling exponent that actually minimises quantization error
    for this specific group of projections.

    A single global alpha is a guess that one exponent suits all 32 blocks.
    It does not: the alpha sweep showed 0.25 beating both 0.5 and 0.125
    overall, which only means 0.25 is the best *compromise*. Here each group
    is scored against the real NF4 quantizer over a small grid and keeps its
    own optimum, with error weighted by the measured activation scale so it
    reflects error the network actually experiences. alpha=0.0 is in the
    grid so a group is free to decline scaling altogether.
    """
    import bitsandbytes.functional as bnbF

    a = act.float().clamp(min=1e-5)
    best, best_err = 0.0, None
    for alpha in grid:
        s = _scale_from(a, alpha).to(weights[0].device)
        err = 0.0
        for W in weights:
            Ws = W.to(torch.bfloat16) * s.to(torch.bfloat16).unsqueeze(0)
            q, st = bnbF.quantize_4bit(Ws, quant_type="nf4")
            W_hat = bnbF.dequantize_4bit(q, st).float() / s.unsqueeze(0)
            err += ((W.float() - W_hat) * a.unsqueeze(0)).pow(2).sum().item()
        if best_err is None or err < best_err:
            best, best_err = alpha, err
    return best


def _apply_awq_scaling(model, acts, alpha=None, scale_down=SCALE_DOWN_PROJ):
    """Fold a per-group scale into the preceding op and out of the weights.
    With alpha=None each group grid-searches its own exponent."""
    chosen = []
    for i, layer in enumerate(model.model.layers):
        groups = [
            (f"{i}.attn_in", layer.input_layernorm,
             [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj], None),
            (f"{i}.mlp_in", layer.post_attention_layernorm,
             [layer.mlp.gate_proj, layer.mlp.up_proj], None),
        ]
        if scale_down:
            groups.append((f"{i}.down_in", None, [layer.mlp.down_proj], layer.mlp.up_proj))

        for key, norm, projs, row_src in groups:
            act = acts[key]
            a = alpha if alpha is not None else _best_alpha([p.weight.data for p in projs], act)
            chosen.append(a)
            s = _scale_from(act, a).to(projs[0].weight.device)
            if norm is not None:
                norm.weight.data /= s.to(norm.weight.dtype)
            else:
                row_src.weight.data /= s.to(row_src.weight.dtype).unsqueeze(1)
            for proj in projs:
                proj.weight.data *= s.to(proj.weight.dtype).unsqueeze(0)

    from collections import Counter
    print(f"per-group alpha chosen: {dict(sorted(Counter(chosen).items()))}")
    return len(chosen)


REPAIR_SECONDS = 240
REPAIR_LR = 1e-4
REPAIR_BATCH = 2
REPAIR_SEQ = 512


def _packed_batches(tokenizer, device, seq_len, batch_size):
    """Stream wikitext-103 train, tokenize, and pack into full fixed-length
    blocks. Packing rather than padding means no token of the budget is
    spent on padding, which matters when the budget is the binding
    constraint."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="train", streaming=True)
    buf, batch = [], []
    for row in ds:
        text = row["text"].strip()
        if len(text) < 200:
            continue
        buf.extend(tokenizer(text, add_special_tokens=False).input_ids)
        while len(buf) >= seq_len:
            batch.append(buf[:seq_len])
            buf = buf[seq_len:]
            if len(batch) == batch_size:
                yield torch.tensor(batch, device=device)
                batch = []


def _repair_locally(model, teacher, tokenizer, neighbours):
    """Distil the deleted layer's neighbours back into agreement with the
    teacher, training ONLY those layers.

    Two ideas make this affordable where a previous session's attempts were
    not. First, the damage is local - one layer was removed, so its
    neighbours are what must absorb its function - and training ~2 layers
    instead of 32 cuts optimizer state and gradient memory enough to afford
    a real batch size. Second, the objective is KL against the teacher's
    full distribution rather than next-token loss on raw text, which is a
    far richer signal per token, and tokens are precisely the scarce
    resource here.
    """
    import bitsandbytes as bnb
    import torch.nn.functional as F

    for p_ in model.parameters():
        p_.requires_grad_(False)
    trainable = []
    for idx in neighbours:
        for p_ in model.model.layers[idx].parameters():
            p_.requires_grad_(True)
            trainable.append(p_)
    n_params = sum(p_.numel() for p_ in trainable)

    device = next(model.parameters()).device
    model.train()
    opt = bnb.optim.PagedAdamW8bit(trainable, lr=REPAIR_LR)

    t0, steps, tokens = time.time(), 0, 0
    for ids in _packed_batches(tokenizer, device, REPAIR_SEQ, REPAIR_BATCH):
        with torch.no_grad():
            t_logits = teacher(ids).logits.float()
        s_logits = model(ids).logits.float()
        t_lp = F.log_softmax(t_logits, dim=-1)
        s_lp = F.log_softmax(s_logits, dim=-1)
        loss = (t_lp.exp() * (t_lp - s_lp)).sum(-1).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        steps += 1
        tokens += ids.numel()
        if time.time() - t0 > REPAIR_SECONDS:
            break

    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    print(f"repair: {steps} steps, {tokens} tokens, {n_params/1e6:.1f}M trainable, "
          f"final KL {loss.item():.4f}")
    return model


def compress(model, tokenizer):
    """Activation-aware scaling with a per-group searched exponent, then
    4-bit NF4 with double quantization."""
    import shutil as _sh
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from prepare import CACHE_DIR

    acts = _norm_input_acts(model, tokenizer)
    n = _apply_awq_scaling(model, acts, alpha=None)
    print(f"awq: rescaled {n} groups with searched alphas, down_proj={SCALE_DOWN_PROJ}")

    tmp = os.path.join(CACHE_DIR, "_awq_tmp")
    _sh.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp, safe_serialization=True)
    tokenizer.save_pretrained(tmp)
    del model
    torch.cuda.empty_cache()

    q = AutoModelForCausalLM.from_pretrained(
        tmp,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["lm_head"],
        ),
        device_map="auto",
    )
    _sh.rmtree(tmp, ignore_errors=True)
    return q


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
