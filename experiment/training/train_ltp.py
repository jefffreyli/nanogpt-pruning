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

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Import the LTP model
from experiment.models.ltp_model import GPTLTP, GPTConfigLTP

# I/O
out_dir = 'out-ltp'
eval_interval = 500
log_interval = 10
eval_iters = 100
eval_only = False
always_save_checkpoint = True
init_from = 'pretrained'  # 'scratch', 'resume', or 'pretrained'

# wandb logging
wandb_log = False
wandb_project = 'nanogpt-ltp'
wandb_run_name = 'ltp-run-' + str(time.time())

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 4
batch_size = 12
block_size = 256

# model - GPT2 small-ish config
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
bias = False

# LTP-specific parameters
# Set prune_mode='learned' to enable token pruning
prune_mode = 'none'  # 'learned' to enable pruning, 'none' to disable
# Threshold for the final layer (higher = more pruning)
final_token_threshold = 0.01
temperature = 5.0  # Temperature for soft masking (Stage 1)
# 'soft' for Stage 1 (learning thresholds), 'hard' for Stage 2 (fine-tuning)
masking_mode = 'soft'
lambda_factor = 0.1  # Regularization weight for pruning loss (Stage 1 only)
freeze_non_threshold_params = False  # If True, only train threshold parameters

# optimizer
learning_rate = 3e-4
max_iters = 10000
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay
decay_lr = True
warmup_iters = 1000
lr_decay_iters = 10000
min_lr = 3e-5

# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available(
) and torch.cuda.is_bf16_supported() else 'float16'
compile = False  # torch.compile not always compatible with custom models

# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith(
    '_') and isinstance(v, (int, float, bool, str))]
# overrides from command line or config file
if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    exec(open(sys.argv[1]).read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# DDP setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * \
    ddp_world_size * batch_size * block_size
if master_process:
    print(f"tokens per iteration: {tokens_per_iter:,}")
    print(f"using device: {device}")
    print(f"using dtype: {dtype}")
    print(f"token pruning enabled: {prune_mode == 'learned'}")
    if prune_mode == 'learned':
        print(f"  masking_mode: {masking_mode}")
        print(f"  final_token_threshold: {final_token_threshold}")
        print(f"  lambda_factor: {lambda_factor}")
        print(f"  freeze non-threshold params: {freeze_non_threshold_params}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32,
           'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
    device_type=device_type, dtype=ptdtype)

# Data loading
data_dir = os.path.join('data', dataset)


def get_batch(split):
    """Load a batch of data from memory-mapped files."""
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'),
                         dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'),
                         dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# Model initialization
iter_num = 0
best_val_loss = 1e9

# Get vocab_size from dataset metadata
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    if master_process:
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# Create model configuration
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=meta_vocab_size if meta_vocab_size is not None else 50304,
    dropout=dropout,
    # LTP-specific parameters
    prune_mode=prune_mode,
    final_token_threshold=final_token_threshold,
    temperature=temperature,
    masking_mode=masking_mode,
    lambda_factor=lambda_factor,
)

if init_from == 'scratch':
    if master_process:
        print("Initializing LTP model from scratch")
    gptconf = GPTConfigLTP(**model_args)
    model = GPTLTP(gptconf)

elif init_from == 'resume':
    if master_process:
        print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']

    # Update model_args with checkpoint values
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]

    # Create model and load state
    gptconf = GPTConfigLTP(**model_args)
    model = GPTLTP(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif init_from == 'pretrained':
    # Load from a pretrained baseline model and add LTP pruning capabilities
    pretrained_path = './experiment/models/baseline_ckpt.pt'  # Update this path as needed
    if master_process:
        print(
            f"Loading pretrained model from {pretrained_path} and adding LTP")

    checkpoint = torch.load(pretrained_path, map_location=device)

    # Get config from checkpoint
    if 'config' in checkpoint:
        ckpt_config = checkpoint['config']
        model_args.update({
            'n_layer': ckpt_config.get('n_layer', n_layer),
            'n_head': ckpt_config.get('n_head', n_head),
            'n_embd': ckpt_config.get('n_embd', n_embd),
            'block_size': ckpt_config.get('block_size', block_size),
            'bias': ckpt_config.get('bias', bias),
        })

    # Get vocab_size from model weights if not in config
    if 'model' in checkpoint:
        vocab_size = checkpoint['model']['transformer.wte.weight'].shape[0]
        model_args['vocab_size'] = vocab_size

    # Create LTP model with pruning enabled
    gptconf = GPTConfigLTP(**model_args)
    model = GPTLTP(gptconf)

    # Load pretrained weights (strict=False allows new pruning parameters)
    if 'model' in checkpoint:
        result = model.load_state_dict(checkpoint['model'], strict=False)
        if master_process:
            print(
                f"Loaded {len(model.state_dict()) - len(result.missing_keys)} pretrained parameters")
            print(
                f"Initialized {len(result.missing_keys)} new pruning parameters")
            if result.missing_keys:
                threshold_params = [
                    k for k in result.missing_keys if 'threshold' in k]
                if threshold_params:
                    print(f"New threshold parameters: {threshold_params}")

model.to(device)

# Optionally freeze all parameters except thresholds
# This is useful for Stage 1 when you want to learn optimal thresholds without modifying pretrained weights
if freeze_non_threshold_params and prune_mode == 'learned':
    if master_process:
        print("Freezing all parameters except threshold parameters...")

    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        if 'threshold' in name:
            param.requires_grad = True
            trainable_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1

    if master_process:
        print(
            f"Frozen {frozen_count} parameter tensors, {trainable_count} threshold parameters trainable")
else:
    if master_process:
        if prune_mode == 'learned':
            print("All parameters trainable (including thresholds)")
        else:
            print("Token pruning disabled - all parameters trainable")

# Initialize GradScaler for mixed precision training
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume' and 'optimizer' in checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

# Compile model (optional)
if compile:
    if master_process:
        print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# Wrap in DDP if needed
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Evaluation function


@torch.no_grad()
def estimate_loss():
    """Estimate loss over multiple batches for train and val sets."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


@torch.no_grad()
def get_pruning_stats():
    """Get pruning statistics for each layer."""
    if prune_mode != 'learned':
        return None

    model.eval()

    # Get underlying model (handle DDP wrapper)
    raw_model = model.module if ddp else model

    # Run a single batch to get pruning info
    X, Y = get_batch('val')
    with ctx:
        # Get pruning stats using the model's built-in method
        stats_list = raw_model.get_pruning_stats(X)

    # Aggregate stats
    stats = {
        'layer_stats': [],
        'total_tokens_kept': 0.0,
        'total_tokens': 0.0,
    }

    for layer_stat in stats_list:
        kept_ratio = layer_stat['keep_ratio']
        if isinstance(kept_ratio, torch.Tensor):
            kept_ratio = kept_ratio.item()

        stats['layer_stats'].append({
            'layer': layer_stat['layer'],
            'kept_ratio': kept_ratio,
            'pruned_ratio': 1.0 - kept_ratio,
            'threshold': layer_stat['threshold'],
            'avg_tokens_kept': layer_stat['avg_tokens_kept'],
        })
        stats['total_tokens_kept'] += kept_ratio
        stats['total_tokens'] += 1.0

    if stats['total_tokens'] > 0:
        stats['avg_kept_ratio'] = stats['total_tokens_kept'] / \
            stats['total_tokens']
        stats['avg_pruned_ratio'] = 1.0 - stats['avg_kept_ratio']
    else:
        stats['avg_kept_ratio'] = 1.0
        stats['avg_pruned_ratio'] = 0.0

    model.train()
    return stats


def print_pruning_stats(stats):
    """Print pruning statistics in a formatted table."""
    if stats is None:
        return

    print("\n" + "="*70)
    print("Token Pruning Statistics per Layer")
    print("="*70)
    print(f"{'Layer':<8} {'Tokens Kept':<14} {'Kept %':<10} {'Pruned %':<10} {'Threshold':<12}")
    print("-"*70)

    for layer_stat in stats['layer_stats']:
        print(f"{layer_stat['layer']:<8} "
              f"{layer_stat['avg_tokens_kept']:<14.1f} "
              f"{layer_stat['kept_ratio']*100:>6.2f}%     "
              f"{layer_stat['pruned_ratio']*100:>6.2f}%     "
              f"{layer_stat['threshold']:.6f}")

    print("-"*70)
    print(f"{'AVERAGE':<8} "
          f"{'':<14} "
          f"{stats['avg_kept_ratio']*100:>6.2f}%     "
          f"{stats['avg_pruned_ratio']*100:>6.2f}%")
    print("="*70 + "\n")

# Learning rate scheduler


def get_lr(it):
    """Cosine learning rate schedule with linear warmup."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# Logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# Training loop
if master_process:
    print(f"\nStarting training for {max_iters} iterations")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(
        f"Effective batch size: {batch_size * gradient_accumulation_steps * ddp_world_size}")
    if prune_mode == 'learned':
        print(f"\nLTP Training Configuration:")
        print(f"  Stage: {masking_mode} pruning")
        print(f"  Final token threshold: {final_token_threshold}")
        if masking_mode == 'soft':
            print(f"  Lambda factor: {lambda_factor}")
            print(f"  Temperature: {temperature}")
        print()

X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

while True:
    # Set learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate and save checkpoint
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Print pruning statistics if token pruning is enabled
        if prune_mode == 'learned':
            pruning_stats = get_pruning_stats()
            print_pruning_stats(pruning_stats)

        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,
            }
            if prune_mode == 'learned' and pruning_stats:
                log_dict.update({
                    "pruning/avg_kept_ratio": pruning_stats['avg_kept_ratio'],
                    "pruning/avg_pruned_ratio": pruning_stats['avg_pruned_ratio'],
                })
                # Log per-layer stats
                for layer_stat in pruning_stats['layer_stats']:
                    log_dict[f"pruning/layer_{layer_stat['layer']}_kept"] = layer_stat['kept_ratio']
                    log_dict[f"pruning/layer_{layer_stat['layer']}_threshold"] = layer_stat['threshold']
            wandb.log(log_dict)

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    if iter_num == 0 and eval_only:
        break

    # Forward-backward pass with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1)

        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps

        # Prefetch next batch
        X, Y = get_batch('train')

        # Backward pass
        scaler.scale(loss).backward()

    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            # Estimate MFU if available
            if hasattr(raw_model, 'estimate_mfu'):
                mfu = raw_model.estimate_mfu(
                    batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    # Termination
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

if master_process:
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final checkpoint saved to {out_dir}/ckpt.pt")
    if prune_mode == 'learned':
        print("\nFinal pruning statistics:")
        final_stats = get_pruning_stats()
        print_pruning_stats(final_stats)
