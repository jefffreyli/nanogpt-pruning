# LTP (Learned Token Pruning) training on Shakespeare character-level dataset

# where to store checkpoints
out_dir = 'out-ltp-shakespeare'

# data
dataset = 'shakespeare_char'

# initialize from scratch (LTP needs to learn pruning thresholds)
init_from = 'scratch'

# logging
wandb_log = False
wandb_project = 'nanogpt-ltp'
wandb_run_name = 'ltp-shakespeare'

# batch / sequence
batch_size = 32
block_size = 256
gradient_accumulation_steps = 4  # 32*4*256 ≈ 32k tokens/iter

# model size (smaller for Shakespeare)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
bias = False

# LTP-specific parameters
use_token_pruning = True
pruning_temperature = 0.01  # Temperature for soft pruning mask
lambda_sparsity = 0.1       # Sparsity regularization weight

# train length / lr
max_iters = 10000
learning_rate = 3e-4
weight_decay = 0.01

# warmup and decay
warmup_iters = 1000
lr_decay_iters = 10000
min_lr = 3e-5
decay_lr = True

# evaluation and logging
eval_interval = 500
log_interval = 10
eval_iters = 100
always_save_checkpoint = True

# optimizer
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# system
compile = False  # torch.compile can be finicky with custom models
