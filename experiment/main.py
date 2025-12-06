import torch
import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment.models.dynamic_token_reduction_model import GPT, GPTConfig
from experiment.models.ltp_model import GPTWithLTP, GPTWithLTPConfig
from experiment.evaluation.evaluation_metrics import EvaluationMetrics

# Load baseline pretrained model checkpoint
checkpoint_path = './experiment/models/baseline_ckpt.pt'
print(f"Loading baseline checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Extract config from checkpoint
ckpt_config = checkpoint['config']

# Get vocab size from checkpoint (GPT-2 uses 50257, not padded 50304)
vocab_size = checkpoint['model']['transformer.wte.weight'].shape[0]

# Create Dynamic Token Reduction model configuration from checkpoint config
config = GPTConfig(
    vocab_size=vocab_size,  # Match checkpoint vocab size
    block_size=ckpt_config['block_size'],
    n_layer=ckpt_config['n_layer'],
    n_head=ckpt_config['n_head'],
    n_embd=ckpt_config['n_embd'],
    dropout=ckpt_config['dropout'],
    bias=ckpt_config['bias'],
    use_token_reduction=True,  # Enable dynamic token reduction
    reduction_layers=(4, 8),   # Apply token reduction at layers 4 and 8
    policy_hidden_dim=256,
    lambda_tokens=1e-4,
    rl_weight=0.1,
)

# Initialize model with pretrained weights
print("Creating Dynamic Token Reduction model...")
dynamic_token_reduction_model = GPT(config)

# Load the pretrained weights (strict=False allows new policy parameters)
result = dynamic_token_reduction_model.load_state_dict(
    checkpoint['model'], strict=False)
print(f"Loaded {len(dynamic_token_reduction_model.state_dict()) - len(result.missing_keys)} pretrained parameters")
print(f"Initialized {len(result.missing_keys)} new policy parameters")
if result.missing_keys:
    print(
        f"New parameters: {[k for k in result.missing_keys if 'policies' in k]}")

print(f"Loaded pretrained weights (iter: {checkpoint.get('iter_num', 'unknown')}, "
      f"val_loss: {checkpoint.get('best_val_loss', 'unknown'):.4f})")
dynamic_token_reduction_model.eval()

# Measure FLOPs
print("\nMeasuring FLOPs for Dynamic Token Reduction model...")
ltp_model_evaluator = EvaluationMetrics(config=config, Model=GPTWithLTP)
baseline_evaluator = EvaluationMetrics()
ltp_model_evaluator.measure_flop_count()
baseline_evaluator.measure_flop_count()
