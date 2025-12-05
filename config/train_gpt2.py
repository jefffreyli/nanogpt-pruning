
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'wikitext2'
wandb_run_name = 'gpt2-124M-wt2'

dataset = 'wikitext2'       # looks in data/wikitext2/
out_dir = 'out-wikitext2'   # where checkpoints/logs will go


# these make a reasonable total batch size for 1 GPU
# 4 batch size * 512 block size * 8 gradaccum * 1 GPU = 16,384 tokens per update
batch_size = 4
block_size = 512
gradient_accumulation_steps = 8

# this makes total number of tokens be
max_iters = 100000
lr_decay_iters = 100000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
