"""
Two-stage training for Hybrid GPT model with Dynamic TR + LTP pruning.

Stage 1 (LTP): Train GPT2 + LTP thresholds (rl_stage = False)
Stage 2 (RL):  Freeze GPT2 + LTP, train only RL policies (rl_stage = True)

Usage:
    export PYTHONPATH="$PWD:$PYTHONPATH"
    
    # Stage 1: LTP training
    python experiment/training/train_hybrid.py config/train_hybrid_wt2.py
    
    # Stage 2: RL training (after modifying config: rl_stage = True, init_from = 'resume')
    python experiment/training/train_hybrid.py config/train_hybrid_wt2.py
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

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from experiment.models.hybrid_model import GPTConfigHybrid, GPTHybrid

# -----------------------------------------------------------------------------
# Default config values
# -----------------------------------------------------------------------------

# I/O
out_dir = 'out-hybrid'
eval_interval = 100
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # 'scratch', 'resume', or 'gpt2*'

# wandb logging
wandb_log = False
wandb_project = 'hybrid-pruning'
wandb_run_name = 'hybrid'

# data
dataset = 'wikitext2'
gradient_accumulation_steps = 8
batch_size = 8
block_size = 512

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = True

# Dynamic TR params
reduction_layers = (4, 8)
policy_hidden_dim = 256
lambda_tokens = 1e-4
rl_weight = 0.1

# LTP params
ltp_layers = (9, 10, 11)
final_token_threshold = 0.01
temperature = 5.0
masking_mode = 'soft'
lambda_factor = 0.1
min_keep_tokens = 64

# Stage control
rl_stage = False  # False = Stage 1 (LTP), True = Stage 2 (RL)

# optimizer
learning_rate = 3e-4
max_iters = 5000
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay
decay_lr = True
warmup_iters = 500
lr_decay_iters = 5000
min_lr = 3e-5

# DDP settings
backend = 'nccl'

# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available(
) and torch.cuda.is_bf16_supported() else 'float16'
compile = False

# -----------------------------------------------------------------------------
# Load config from command line
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith(
    '_') and isinstance(v, (int, float, bool, str, tuple))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# DDP setup
# -----------------------------------------------------------------------------
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
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
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

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

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
data_dir = os.path.join('data', dataset)


def get_batch(split):
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


# -----------------------------------------------------------------------------
# Vocab size from meta.pkl
# -----------------------------------------------------------------------------
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    if master_process:
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# -----------------------------------------------------------------------------
# Model initialization
# -----------------------------------------------------------------------------
iter_num = 0
best_val_loss = 1e9

# Training history for plotting
training_history = {
    'train_losses': [],
    'val_losses': [],
    'val_perplexities': [],
}

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=meta_vocab_size if meta_vocab_size is not None else 50304,
    dropout=dropout,
    # Dynamic TR params
    reduction_layers=tuple(reduction_layers) if isinstance(
        reduction_layers, list) else reduction_layers,
    policy_hidden_dim=policy_hidden_dim,
    lambda_tokens=lambda_tokens,
    rl_weight=rl_weight,
    # LTP params
    ltp_layers=tuple(ltp_layers) if isinstance(
        ltp_layers, list) else ltp_layers,
    final_token_threshold=final_token_threshold,
    temperature=temperature,
    masking_mode=masking_mode,
    lambda_factor=lambda_factor,
    min_keep_tokens=min_keep_tokens,
)

if master_process:
    print(f"Stage: {'RL (Stage 2)' if rl_stage else 'LTP (Stage 1)'}")
    print(f"Creating hybrid model with config: {model_args}")

if init_from == 'scratch':
    if master_process:
        print("Initializing a new model from scratch")
    gptconf = GPTConfigHybrid(**model_args)
    model = GPTHybrid(gptconf)

elif init_from == 'resume':
    if master_process:
        print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']

    # Force these config attributes to be equal
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]

    gptconf = GPTConfigHybrid(**model_args)
    model = GPTHybrid(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if master_process:
        if missing:
            print(f"Missing keys when loading: {missing}")
        if unexpected:
            print(f"Unexpected keys when loading: {unexpected}")

    if rl_stage:
        # Stage 2: start fresh iteration count
        iter_num = 0
        best_val_loss = 1e9
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'val_perplexities': [],
        }
        if master_process:
            print("RL stage: resetting iteration count and training history")
    else:
        # Normal resume for Stage 1
        iter_num = checkpoint.get('iter_num', 0)
        best_val_loss = checkpoint.get('best_val_loss', 1e9)
        if 'training_history' in checkpoint:
            training_history = checkpoint['training_history']
            if master_process:
                print(
                    f"Loaded training history with {len(training_history['train_losses'])} training steps")

elif init_from.startswith('gpt2'):
    if master_process:
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # Note: This would require custom loading logic for hybrid model
    # For now, we start from scratch and let users load baseline weights
    gptconf = GPTConfigHybrid(**model_args)
    model = GPTHybrid(gptconf)

# Crop block size if needed
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

# -----------------------------------------------------------------------------
# Freeze parameters for Stage 2 (RL training)
# -----------------------------------------------------------------------------
raw_model = model

if rl_stage:
    if master_process:
        print("RL stage: freezing GPT2 backbone and LTP thresholds, training only policies")

    trainable_params = []
    frozen_params = []

    for name, param in raw_model.named_parameters():
        if name.startswith("policies."):
            param.requires_grad = True
            trainable_params.append(name)
        else:
            param.requires_grad = False
            frozen_params.append(name)

    if master_process:
        print(f"Trainable parameters ({len(trainable_params)}):")
        for n in trainable_params:
            print(f"  {n}")
        print(f"Frozen parameters count: {len(frozen_params)}")

# -----------------------------------------------------------------------------
# Optimizer and GradScaler
# -----------------------------------------------------------------------------
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = raw_model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type)

# Load optimizer state only for Stage 1 resume
if init_from == 'resume' and not rl_stage:
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

checkpoint = None  # free memory

# -----------------------------------------------------------------------------
# Compile model
# -----------------------------------------------------------------------------
if compile:
    if master_process:
        print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# Wrap in DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model


# -----------------------------------------------------------------------------
# Loss computation with RL
# -----------------------------------------------------------------------------
def compute_total_loss(lm_loss, rl_info, seq_len, raw_model):
    """
    Combine LM loss with RL policy gradient loss.
    """
    policy_logprobs = rl_info["policy_logprobs"]
    num_selected_tokens = rl_info["num_selected_tokens"]

    if len(policy_logprobs) == 0:
        return lm_loss, None

    logp = torch.stack(policy_logprobs, dim=0)
    selected = torch.stack(num_selected_tokens, dim=0)

    L = logp.size(0)
    T = seq_len

    logp_total = logp.sum(dim=0) / (L * T)
    frac_selected = selected.sum(dim=0) / (L * T)
    avg_frac = frac_selected.mean().item()

    lam = getattr(raw_model.config, "lambda_tokens", lambda_tokens)
    alpha = getattr(raw_model.config, "rl_weight", rl_weight)

    with torch.no_grad():
        base = -lm_loss.detach()
        base = base.expand_as(frac_selected)
        reward = base - lam * frac_selected
        reward = reward - reward.mean()

    policy_loss = -(reward * logp_total).mean()
    total_loss = lm_loss + alpha * policy_loss

    return total_loss, avg_frac


# -----------------------------------------------------------------------------
# Learning rate schedule
# -----------------------------------------------------------------------------
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                # Evaluate without token reduction for fair comparison
                logits, loss, _ = raw_model(
                    X, Y, use_token_reduction=False, policy_training=False)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# -----------------------------------------------------------------------------
# Wandb logging
# -----------------------------------------------------------------------------
if wandb_log and master_process:
    import wandb
    run_name = f"{wandb_run_name}-{'rl' if rl_stage else 'ltp'}"
    wandb.init(project=wandb_project, name=run_name, config=config)


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0

while True:
    # Set learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluation and checkpointing
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        val_loss = losses['val'].item() if torch.is_tensor(
            losses['val']) else losses['val']
        val_perplexity = math.exp(val_loss)

        # Log to training history
        training_history['val_losses'].append((iter_num, val_loss))
        training_history['val_perplexities'].append((iter_num, val_perplexity))

        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {val_loss:.4f}, val ppl {val_perplexity:.2f}")

        # Print pruning stats
        X_sample, _ = get_batch('val')
        stats = raw_model.get_pruning_stats(X_sample[:1])
        print("\nPruning Statistics:")
        print("Layer    Type        Tokens Kept    Kept %")
        print("-" * 50)
        for s in stats:
            layer_type = s.get('type', 'Standard')
            if layer_type in ['LTP', 'DynamicTR']:
                threshold_str = f"  (threshold: {s.get('threshold', 'N/A'):.6f})" if 'threshold' in s else ""
                print(
                    f"{s['layer']:2d}       {layer_type:10s}  {s['avg_tokens_kept']:6.1f}       {s['keep_ratio']*100:6.2f}%{threshold_str}")
        print()

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": val_loss,
                "val/perplexity": val_perplexity,
                "lr": lr,
                "mfu": running_mfu * 100,
            })

        if val_loss < best_val_loss or always_save_checkpoint:
            best_val_loss = min(best_val_loss, val_loss)
            if iter_num > 0:
                ckpt = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'training_history': training_history,
                }
                ckpt_name = 'rl_ckpt.pt' if rl_stage else 'ckpt.pt'
                print(f"saving checkpoint to {out_dir}/{ckpt_name}")
                torch.save(ckpt, os.path.join(out_dir, ckpt_name))

    if iter_num == 0 and eval_only:
        break

    # Forward/backward with gradient accumulation
    avg_frac = None
    optimizer.zero_grad(set_to_none=True)

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1)

        with ctx:
            if rl_stage:
                # Stage 2: use Dynamic TR with RL training
                logits, lm_loss, rl_info = raw_model(
                    X, Y,
                    use_token_reduction=True,
                    policy_training=True,
                )
                total_loss, avg_frac_step = compute_total_loss(
                    lm_loss, rl_info, X.size(1), raw_model)
                avg_frac = avg_frac_step
                loss = total_loss / gradient_accumulation_steps
            else:
                # Stage 1: LTP training (no Dynamic TR)
                logits, loss, _ = raw_model(
                    X, Y, use_token_reduction=False, policy_training=False)
                loss = loss / gradient_accumulation_steps

        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        training_history['train_losses'].append((iter_num, lossf))

        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(
                batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

        if rl_stage and avg_frac is not None:
            print(
                f"iter {iter_num}: loss {lossf:.4f}, kept_frac ~{avg_frac:.3f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        else:
            print(
                f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
