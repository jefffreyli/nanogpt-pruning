# Vocabulary Pruning training on WikiText-2
# This config trains a GPT-2 model with dynamic vocabulary pruning

# where to store checkpoints
out_dir = "out-vocab-pruning-wt2"

# data
dataset = "wikitext2"

# Initialize from pretrained GPT-2 baseline
# Options:
#   'scratch'    - train from random initialization
#   'resume'     - resume from existing checkpoint
#   'gpt2*'      - initialize from pretrained GPT-2 (e.g., 'gpt2', 'gpt2-medium')
init_from = "gpt2"

# logging
wandb_log = False
wandb_project = "wikitext2"
wandb_run_name = "vocab-pruning-wt2"

# batch / sequence
batch_size = 8       # micro-batch size
block_size = 512
gradient_accumulation_steps = 8  # 8*8*512 ≈ 32k tokens per step

# model size (matches gpt2 small)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = True

# --- Dynamic Vocabulary Pruning Config ---
# Layer to use for determining active vocabulary (0-indexed)
pruning_layer = 3
pruning_topk = 500      # Number of tokens to keep active
aux_loss_weight = 0.1   # Weight for the auxiliary loss on the pruning layer.
# Set to 0.0 to disable. 0.1-0.3 is usually good for training.

# training schedule
learning_rate = 3e-4
max_iters = 10000
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
eval_only = False

# system
device = "cuda"  # 'cpu', 'cuda', 'cuda:0', etc., or 'mps' on macbooks
dtype = "bfloat16"  # 'float32', 'bfloat16', or 'float16'
compile = False  # torch.compile can be finicky with custom models
backend = "nccl"  # for DDP
