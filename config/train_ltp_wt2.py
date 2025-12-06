# LTP (Learned Token Pruning) training on WikiText-2
# This config trains LTP from a pretrained GPT-2 checkpoint

# where to store checkpoints
out_dir = 'out-ltp-wt2'

# data
dataset = 'wikitext2'

# Initialize from pretrained GPT-2 baseline and add LTP pruning
# Options:
#   'pretrained' - Load baseline_ckpt.pt and add pruning parameters (recommended)
#   'scratch'    - Train LTP model from random initialization
#   'resume'     - Resume from existing LTP checkpoint
init_from = 'pretrained'

# logging
wandb_log = False
wandb_project = 'nanogpt-ltp'
wandb_run_name = 'ltp-wt2'

# batch / sequence (match baseline)
batch_size = 8
block_size = 512
gradient_accumulation_steps = 8  # 8*8*512 ≈ 32k tokens/iter

# model size (GPT-2 small)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = True

# LTP-specific parameters
use_token_pruning = True
pruning_temperature = 0.01  # Temperature for soft pruning mask
lambda_sparsity = 0.1       # Sparsity regularization weight
# Importance scoring method for causal attention
# Options: 'naive_col' (original, biased), 'row', 'causal_col' (recommended), 'future_aware'
importance_method = 'causal_col'

# train length / lr
max_iters = 1000
learning_rate = 1e-4
weight_decay = 0.1

# warmup and decay
warmup_iters = 100
lr_decay_iters = 1000
min_lr = 3e-5
decay_lr = True

# evaluation and logging
eval_interval = 100
log_interval = 10
eval_iters = 200
always_save_checkpoint = True

# optimizer
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# system
compile = False  # torch.compile can be finicky with custom models
