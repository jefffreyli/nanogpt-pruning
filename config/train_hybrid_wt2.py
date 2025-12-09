# Hybrid Model Training on WikiText-2
# Combines Dynamic Token Reduction (RL) in early layers with LTP in later layers
#
# Two-stage training:
#   Stage 1 (LTP): rl_stage = False, init_from = 'scratch' or 'resume'
#   Stage 2 (RL):  rl_stage = True,  init_from = 'resume'
#
# Usage:
#   export PYTHONPATH="$PWD:$PYTHONPATH"
#
#   # Stage 1: Train LTP with unfrozen GPT2
#   python experiment/training/train_hybrid.py config/train_hybrid_wt2.py
#
#   # Stage 2: Freeze GPT2 + LTP, train RL policies
#   # (First edit this file: set rl_stage = True and init_from = 'resume')
#   python experiment/training/train_hybrid.py config/train_hybrid_wt2.py

# -----------------------------------------------------------------------------
# Output directory
# -----------------------------------------------------------------------------
out_dir = 'out-hybrid-wt2'

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
dataset = 'wikitext2'

# -----------------------------------------------------------------------------
# Stage control - EDIT THIS FOR STAGE 2
# -----------------------------------------------------------------------------
# Stage 1: rl_stage = False, trains GPT2 + LTP thresholds
# Stage 2: rl_stage = True,  freezes GPT2 + LTP, trains only RL policies
rl_stage = False

# For Stage 1: use 'scratch' or 'resume' (to continue Stage 1 training)
# For Stage 2: use 'resume' (loads Stage 1 checkpoint)
init_from = 'scratch'

# -----------------------------------------------------------------------------
# Model architecture (GPT-2 small)
# -----------------------------------------------------------------------------
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = True

# -----------------------------------------------------------------------------
# Dynamic Token Reduction (RL) - applied in EARLY layers
# -----------------------------------------------------------------------------
reduction_layers = (4, 8)       # Layers where RL policies prune tokens
policy_hidden_dim = 256         # Hidden dim for policy network
lambda_tokens = 1e-4            # Penalty on number of selected tokens
rl_weight = 0.1                 # Weight of RL loss vs LM loss

# -----------------------------------------------------------------------------
# Learned Token Pruning (LTP) - applied in LATER layers
# -----------------------------------------------------------------------------
ltp_layers = (9, 10, 11)        # Layers where LTP prunes tokens
final_token_threshold = 0.01   # Global scale for per-layer thresholds
temperature = 5.0              # Temperature for soft pruning mask
masking_mode = 'soft'          # 'soft' for training, 'hard' for inference
lambda_factor = 0.1            # Sparsity regularization weight
# Last N tokens never pruned (for LM loss stability)
min_keep_tokens = 64

# -----------------------------------------------------------------------------
# Batch and sequence configuration
# -----------------------------------------------------------------------------
batch_size = 8
block_size = 512
gradient_accumulation_steps = 8  # Effective batch: 8 * 8 * 512 = 32k tokens

# -----------------------------------------------------------------------------
# Training schedule
# -----------------------------------------------------------------------------
# Stage 1 (LTP): longer training with higher LR
# Stage 2 (RL):  shorter fine-tuning with lower LR
if rl_stage:
    max_iters = 300
    learning_rate = 1e-4
    warmup_iters = 30
else:
    max_iters = 300
    learning_rate = 3e-4
    warmup_iters = 30

lr_decay_iters = max_iters
min_lr = learning_rate / 10

weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True

# -----------------------------------------------------------------------------
# Evaluation and logging
# -----------------------------------------------------------------------------
eval_interval = 50
log_interval = 10
eval_iters = 200
always_save_checkpoint = True

# -----------------------------------------------------------------------------
# Wandb (optional)
# -----------------------------------------------------------------------------
wandb_log = False
wandb_project = 'hybrid-pruning'
wandb_run_name = 'hybrid-wt2'

# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------
compile = False  # torch.compile can be finicky with custom models
