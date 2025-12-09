# Baseline GPT-2 fine-tuning on WikiText-2 (no RL, no token reduction)

# where to store checkpoints
out_dir = 'out-wt2-baseline'

# data
dataset = 'wikitext2'

# initialize from pretrained GPT-2 small
init_from = 'gpt2'   # uses GPT.from_pretrained

# logging
wandb_log = False
wandb_project = 'wikitext2'
wandb_run_name = 'gpt2-wt2-baseline'

# batch / sequence
batch_size = 8       # micro-batch size
block_size = 512
gradient_accumulation_steps = 8  # 8*8*512 ≈ 32k tokens/iter

# model size (matches gpt2 small)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = True

# train length / lr
max_iters = 300
learning_rate = 3e-4

# I'd shorten warmup; 1000 steps of warmup on a 1000-step run is overkill.
warmup_iters = 100

lr_decay_iters = max_iters
min_lr = 3e-5

# (optional but useful) more frequent evals for a small run
eval_interval = 100
log_interval = 10

# For first run, I'd turn compile off to keep debugging easier; you can turn it back on later.
compile = False

# RL-related flags OFF
use_token_reduction = False
rl_stage = False
lambda_tokens = 1e-4
rl_weight = 0.1
