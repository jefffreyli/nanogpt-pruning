# LTP (Learned Token Pruning) training on WikiText-2
# This config trains LTP from a pretrained GPT-2 checkpoint.

# where to store checkpoints
out_dir = "out-ltp-wt2"

# data
dataset = "wikitext2"

# Initialize from pretrained GPT-2 baseline and add LTP pruning
# Options:
#   'pretrained' - load baseline_ckpt.pt and add pruning parameters
#   'scratch'    - train LTP model from random initialization
#   'resume'     - resume from existing LTP checkpoint
init_from = "pretrained"

# batch / sequence
batch_size = 8
block_size = 512
gradient_accumulation_steps = 8  # 8*8*512 ≈ 32k tokens per step

# GPT-2 small
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = True

# LTP-specific knobs (high-level)
use_token_pruning = True
pruning_temperature = 5.0    # temperature for soft pruning mask
lambda_sparsity = 0.1        # sparsity regularization weight

# how aggressively to protect the tail
protected_tail_tokens = 128   # last 128 tokens never pruned

# training schedule
learning_rate = 3e-4
max_iters = 300
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# LR decay
decay_lr = True
warmup_iters = 1000
lr_decay_iters = max_iters
min_lr = 3e-5

# eval & logging
eval_interval = 100
log_interval = 10
eval_iters = 200
always_save_checkpoint = True

# system
compile = False  # torch.compile can be finicky with custom models


# ---------------------------------------------------------------------------
# Map config variables to what train_ltp.py / GPTConfigLTP expect
# ---------------------------------------------------------------------------

# Enable or disable pruning
prune_mode = "learned" if use_token_pruning else "none"

# Stage 1 vs Stage 2:
#   - Stage 1: masking_mode = 'soft'  (learn thresholds with soft masks)
#   - Stage 2: set masking_mode = 'hard' and init_from = 'resume'
masking_mode = "soft"

# Temperature and sparsity regularization for soft masks
temperature = pruning_temperature
lambda_factor = lambda_sparsity

# Global threshold scale (used by per-layer learnable thresholds)
final_token_threshold = 0.01

# Number of tail tokens that are never pruned (LM loss stability)
min_keep_tokens = protected_tail_tokens

freeze_base_weights = False

