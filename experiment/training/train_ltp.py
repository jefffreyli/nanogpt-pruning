"""
Full training loop for LTP model with Learned Token Pruning.

Usage:
    export PYTHONPATH="$PWD:$PYTHONPATH"
    python experiment/training/train_ltp.py config/train_ltp_wt2.py
"""

import os
import sys
import time
import math
import pickle
from contextlib import nullcontext
import importlib

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from experiment.models.ltp_model import GPTLTP, GPTConfigLTP

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

if len(sys.argv) < 2:
    print("Usage: train_ltp.py <config_module>")
    sys.exit(1)

config_module = sys.argv[1]
if config_module.endswith(".py"):
    config_module = config_module[:-3]
if config_module.startswith("config."):
    config_module = config_module
else:
    config_module = f"config.{config_module}"

cfg = importlib.import_module(config_module)

# This assumes your config defines these:
out_dir = cfg.out_dir
eval_interval = getattr(cfg, "eval_interval", 100)
log_interval = getattr(cfg, "log_interval", 10)
eval_iters = getattr(cfg, "eval_iters", 200)
always_save_checkpoint = getattr(cfg, "always_save_checkpoint", True)

dataset = cfg.dataset
gradient_accumulation_steps = cfg.gradient_accumulation_steps
batch_size = cfg.batch_size
block_size = cfg.block_size

n_layer = cfg.n_layer
n_head = cfg.n_head
n_embd = cfg.n_embd
dropout = cfg.dropout
bias = cfg.bias

# LTP-specific
prune_mode = getattr(cfg, "prune_mode", "learned")
final_token_threshold = getattr(cfg, "final_token_threshold", 0.01)
temperature = getattr(cfg, "temperature", 5.0)
masking_mode = getattr(cfg, "masking_mode", "soft")
lambda_factor = getattr(cfg, "lambda_factor", 0.1)
min_keep_tokens = getattr(cfg, "min_keep_tokens", 64)

# optimizer hyperparams
learning_rate = getattr(cfg, "learning_rate", 3e-4)
max_iters = getattr(cfg, "max_iters", 10000)
weight_decay = getattr(cfg, "weight_decay", 0.01)
beta1 = getattr(cfg, "beta1", 0.9)
beta2 = getattr(cfg, "beta2", 0.95)
grad_clip = getattr(cfg, "grad_clip", 1.0)

# lr decay
decay_lr = getattr(cfg, "decay_lr", True)
warmup_iters = getattr(cfg, "warmup_iters", 1000)
lr_decay_iters = getattr(cfg, "lr_decay_iters", max_iters)
min_lr = getattr(cfg, "min_lr", 1e-5)

compile = getattr(cfg, "compile", False)

# DDP setup ------------------------------------------------------------------

ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ddp_rank = 0
    ddp_world_size = 1
    master_process = True

device_type = "cuda" if "cuda" in device else "cpu"
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(
    device_type=device_type, dtype=torch.bfloat16)

# ---------------------------------------------------------------------------
# Data loading (NanoGPT-style .bin)
# ---------------------------------------------------------------------------

data_dir = os.path.join("data", dataset)
train_data = np.memmap(os.path.join(
    data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"),
                     dtype=np.uint16, mode="r")


def get_batch(split: str):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i: i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(
        data[i + 1: i + 1 + block_size].astype(np.int64)) for i in ix])
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y

# meta.pkl for vocab size ----------------------------------------------------


meta_vocab_size = None
meta_path = os.path.join(data_dir, "meta.pkl")
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    if master_process:
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# ---------------------------------------------------------------------------
# Model init
# ---------------------------------------------------------------------------

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=meta_vocab_size if meta_vocab_size is not None else 50304,
    dropout=dropout,
    # LTP-specific
    prune_mode=prune_mode,
    final_token_threshold=final_token_threshold,
    temperature=temperature,
    masking_mode=masking_mode,
    lambda_factor=lambda_factor,
    min_keep_tokens=min_keep_tokens,
)

if master_process:
    print("creating LTP model with config:", model_args)

gptconf = GPTConfigLTP(**model_args)
model = GPTLTP(gptconf)

# Check if we should resume from previous LTP training
resume_from_ltp = getattr(cfg, "resume_from_ltp", False)
if resume_from_ltp and os.path.exists(os.path.join(out_dir, "ckpt.pt")):
    if master_process:
        print(f"Resuming LTP training from {out_dir}/ckpt.pt")
    ckpt = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
else:
    # Load from baseline checkpoint
    ckpt = torch.load("experiment/models/baseline_ckpt.pt",
                      map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)

if compile and device_type == "cuda":
    model = torch.compile(model)


model.to(device)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# --- NEW: optionally freeze all base weights except pruning thresholds ---
freeze_base = getattr(cfg, "freeze_base_weights", False)

# If using DDP later, we'll choose the correct raw_model below
raw_model_for_freeze = model

# If you wrap with DDP *before* this block, do:
# raw_model_for_freeze = model.module if isinstance(model, DDP) else model

if freeze_base:
    print("Freezing all parameters except LTP thresholds...")
    trainable = []
    frozen = []
    for name, param in raw_model_for_freeze.named_parameters():
        if "threshold" in name:
            param.requires_grad = True
            trainable.append(name)
        else:
            param.requires_grad = False
            frozen.append(name)
    print(f"Trainable parameters (should be only thresholds):")
    for n in trainable:
        print("  ", n)
    print(f"Frozen parameters count: {len(frozen)}")
# --- END NEW ---
# optimizer ------------------------------------------------------------------

raw_model = model.module if ddp else model
optimizer = raw_model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type)

# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------


def get_lr(it):
    if not decay_lr:
        return learning_rate
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def estimate_loss():
    out = {}
    raw = model.module if ddp else model
    raw.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = raw(X, Y)
            losses[k] = loss.detach()
        out[split] = losses.mean().item()
    raw.train()
    return out

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


if master_process:
    os.makedirs(out_dir, exist_ok=True)

iter_num = 0
best_val_loss = 1e9
tokens_per_iter = gradient_accumulation_steps * \
    batch_size * block_size * ddp_world_size

# Initialize training history for plotting
training_history = {
    'train_losses': [],      # list of (iter_num, loss)
    'val_losses': [],        # list of (iter_num, loss)
    'val_perplexities': [],  # list of (iter_num, perplexity)
}

# Load training state if resuming
if resume_from_ltp and os.path.exists(os.path.join(out_dir, "ckpt.pt")):
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    if 'iter_num' in checkpoint:
        iter_num = checkpoint['iter_num']
    if 'best_val_loss' in checkpoint:
        best_val_loss = checkpoint['best_val_loss']
    if 'training_history' in checkpoint:
        training_history = checkpoint['training_history']
        if master_process:
            print(
                f"Loaded training history with {len(training_history['train_losses'])} training steps")
    if master_process:
        print(
            f"Resuming from iter {iter_num} with best_val_loss {best_val_loss:.4f}")

raw_model = model.module if ddp else model
raw_model.train()

while True:
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        val_perplexity = math.exp(losses['val'])

        # Log validation metrics to history
        training_history['val_losses'].append((iter_num, losses['val']))
        training_history['val_perplexities'].append((iter_num, val_perplexity))

        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}, val ppl {val_perplexity:.2f}"
        )
        # optional: print pruning stats on a small sample
        X, _ = get_batch("val")
        stats = raw_model.get_pruning_stats(X[:1])
        print("Token Pruning Statistics per Layer")
        print("Layer    Tokens Kept    Kept %     Pruned %   Threshold")
        for s in stats:
            kept = s["avg_tokens_kept"]
            keep_ratio = s["keep_ratio"] * 100.0
            pruned_ratio = 100.0 - keep_ratio
            print(
                f"{s['layer']:2d}        {kept:6.1f}       {keep_ratio:6.2f}%    "
                f"{pruned_ratio:6.2f}%     {s['threshold']:.6f}"
            )

        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = min(best_val_loss, losses["val"])
            if master_process:
                ckpt_path = os.path.join(out_dir, "ckpt.pt")
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "model_args": model_args,
                    "training_history": training_history,
                }
                torch.save(checkpoint, ckpt_path)
                print(f"saved checkpoint to {ckpt_path}")

    if iter_num > max_iters:
        break

    # gradient accumulation
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for micro in range(gradient_accumulation_steps):
        X, Y = get_batch("train")
        with ctx:
            logits, loss = raw_model(X, Y)
            loss = loss / gradient_accumulation_steps
        loss.backward()
        total_loss += loss.item()

    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    # Log training loss to history
    if master_process:
        training_history['train_losses'].append((iter_num, total_loss))

    iter_num += 1

destroy_process_group() if ddp else None
