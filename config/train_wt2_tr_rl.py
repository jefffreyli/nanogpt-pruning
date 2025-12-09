# RL stage: dynamic token reduction on top of WT2 baseline

out_dir = 'out-wt2-tr-rl'      # new dir for RL checkpoints

dataset = 'wikitext2'

# resume from the baseline checkpoint
init_from = 'resume'           # train_dynamic_tr.py will load out_dir/ckpt.pt
# (before running, copy baseline ckpt into this new out_dir)

wandb_log = False
wandb_project = 'wikitext2'
wandb_run_name = 'gpt2-wt2-tr-rl'

batch_size = 8
block_size = 512
gradient_accumulation_steps = 8

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = True

max_iters = 300          # fewer steps for RL fine-tuning
learning_rate = 1e-4     # smaller LR for policy + slight LM tuning
warmup_iters = 50
lr_decay_iters = max_iters
min_lr = 1e-5

eval_interval = 100
log_interval = 10
compile = False

# RL / token reduction
use_token_reduction = True
rl_stage = True
lambda_tokens = 1e-4     # penalty on #tokens
rl_weight = 0.1          # how strong the RL term is vs LM loss
