# Hybrid RL+LTP Model Training on WikiText-2
# Loads a trained RL model and adds LTP to the last 3 layers
#
# This config is for training ONLY the LTP thresholds while keeping:
#   - GPT2 backbone frozen
#   - RL policies frozen (already trained)
#   - LTP layers trainable
#
# Workflow:
#   1. First convert your RL checkpoint:
#      python experiment/training/convert_rl_to_hybrid.py \
#          --rl_checkpoint out-dynamic-tr-wt2/ckpt.pt \
#          --output out-hybrid-rl-ltp/converted_ckpt.pt
#
#   2. Then train LTP layers:
#      export PYTHONPATH="$PWD:$PYTHONPATH"
#      python experiment/training/train_hybrid.py config/train_hybrid_rl_ltp.py

# -----------------------------------------------------------------------------
# Output directory
# -----------------------------------------------------------------------------
out_dir = 'out-hybrid-rl-ltp'

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
dataset = 'wikitext2'

# -----------------------------------------------------------------------------
# Initialization - Load converted checkpoint
# -----------------------------------------------------------------------------
# Use 'resume' to load the converted checkpoint from out-hybrid-rl-ltp/converted_ckpt.pt
# Or use 'rl_checkpoint' to indicate we're loading from a converted RL model
init_from = 'resume'

# -----------------------------------------------------------------------------
# Training mode - LTP training only
# -----------------------------------------------------------------------------
# rl_stage = False means we train LTP (not RL policies)
# The training script will freeze RL policies and GPT2 backbone
rl_stage = False
ltp_training_only = True  # Special flag to indicate LTP-only training

# -----------------------------------------------------------------------------
# Model architecture (must match your trained RL model)
# -----------------------------------------------------------------------------
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = True

# -----------------------------------------------------------------------------
# Dynamic Token Reduction (RL) - FROZEN, from trained model
# -----------------------------------------------------------------------------
# Layers where RL policies prune tokens (frozen)
reduction_layers = (4, 8)
policy_hidden_dim = 256         # Hidden dim for policy network
lambda_tokens = 1e-5            # Not used during LTP training
rl_weight = 0.01                # Not used during LTP training

# -----------------------------------------------------------------------------
# Learned Token Pruning (LTP) - TRAINABLE on last 3 layers
# -----------------------------------------------------------------------------
ltp_layers = (9, 10, 11)        # Last 3 layers use LTP (trainable)
final_token_threshold = 0.01    # Global scale for per-layer thresholds
temperature = 5.0               # Temperature for soft pruning mask
masking_mode = 'soft'           # 'soft' for training, 'hard' for inference
lambda_factor = 0.1             # Sparsity regularization weight
min_keep_tokens = 64            # Last N tokens never pruned

# -----------------------------------------------------------------------------
# Batch and sequence configuration
# -----------------------------------------------------------------------------
batch_size = 8
block_size = 512
gradient_accumulation_steps = 8  # Effective batch: 8 * 8 * 512 = 32k tokens

# -----------------------------------------------------------------------------
# Training schedule - LTP training
# -----------------------------------------------------------------------------
# Train only LTP thresholds, so we can use moderate learning rate
max_iters = 2000                 # Longer training for LTP convergence
learning_rate = 1e-4             # Moderate LR for threshold learning
warmup_iters = 100               # Gradual warmup
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
eval_interval = 100              # Evaluate every 100 iterations
log_interval = 10                # Log every 10 iterations
eval_iters = 200                 # Number of iterations for evaluation
always_save_checkpoint = True    # Save checkpoint after each evaluation

# -----------------------------------------------------------------------------
# Wandb (optional)
# -----------------------------------------------------------------------------
wandb_log = False
wandb_project = 'hybrid-pruning'
wandb_run_name = 'hybrid-rl-ltp'

# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------
compile = False  # torch.compile can be finicky with custom models
