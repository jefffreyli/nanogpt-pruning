"""
Evaluation script to measure FLOPs and perplexity for different GPT model variants:
1. Baseline GPT model
2. LTP (Learned Token Pruning) model
3. Dynamic Token Reduction model (TR-BERT style with RL)
"""

import torch
import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment.evaluation.evaluation_metrics import EvaluationMetrics
from experiment.models.dynamic_token_reduction_model import GPT as DynamicTRGPT, GPTConfig as DynamicTRGPTConfig
from experiment.models.ltp_model import GPTWithLTP, GPTWithLTPConfig
from model import GPT as BaselineGPT, GPTConfig as BaselineGPTConfig

# Import models and configs


def evaluate_baseline_model():
    """Evaluate the baseline GPT model (no token reduction)."""
    print("=" * 60)
    print("BASELINE GPT MODEL")
    print("=" * 60)

    # Create baseline config (small model for testing)
    config = BaselineGPTConfig(
        vocab_size=50304,
        block_size=256,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.0,
        bias=True
    )

    # Create evaluator with baseline model
    evaluator = EvaluationMetrics(config=config, Model=BaselineGPT)

    print("\n--- FLOPs Measurement ---")
    macs, params = evaluator.measure_flop_count()

    print("\n--- Perplexity Measurement ---")
    loss, perplexity = evaluator.measure_perplexity()

    return {
        'evaluator': evaluator,
        'macs': macs,
        'params': params,
        'loss': loss,
        'perplexity': perplexity
    }


def evaluate_ltp_model(use_token_pruning=False):
    """Evaluate the LTP (Learned Token Pruning) model."""
    print("\n" + "=" * 60)
    pruning_status = "WITH" if use_token_pruning else "WITHOUT"
    print(f"LTP MODEL ({pruning_status} TOKEN PRUNING)")
    print("=" * 60)

    # Create LTP config
    config = GPTWithLTPConfig(
        vocab_size=50304,
        block_size=256,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.0,
        bias=True,
        use_token_pruning=use_token_pruning,
        pruning_temperature=0.01,
        lambda_sparsity=0.1
    )

    # Create evaluator with LTP model
    evaluator = EvaluationMetrics(config=config, Model=GPTWithLTP)

    print("\n--- FLOPs Measurement ---")
    macs, params = evaluator.measure_flop_count()

    print("\n--- Perplexity Measurement ---")
    loss, perplexity = evaluator.measure_perplexity()

    return {
        'evaluator': evaluator,
        'macs': macs,
        'params': params,
        'loss': loss,
        'perplexity': perplexity
    }


def evaluate_dynamic_tr_model(use_token_reduction=False):
    """Evaluate the Dynamic Token Reduction model (TR-BERT style with RL)."""
    print("\n" + "=" * 60)
    reduction_status = "WITH" if use_token_reduction else "WITHOUT"
    print(
        f"DYNAMIC TOKEN REDUCTION MODEL ({reduction_status} TOKEN REDUCTION)")
    print("=" * 60)

    # Create Dynamic TR config
    config = DynamicTRGPTConfig(
        vocab_size=50304,
        block_size=256,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.0,
        bias=True,
        use_token_reduction=use_token_reduction,
        reduction_layers=(2, 4),  # Apply token reduction at layers 2 and 4
        policy_hidden_dim=256,
        lambda_tokens=1e-4,
        rl_weight=0.1
    )

    # Create evaluator with Dynamic TR model
    evaluator = EvaluationMetrics(config=config, Model=DynamicTRGPT)

    print("\n--- FLOPs Measurement ---")
    macs, params = evaluator.measure_flop_count()

    print("\n--- Perplexity Measurement ---")
    loss, perplexity = evaluator.measure_perplexity()

    return {
        'evaluator': evaluator,
        'macs': macs,
        'params': params,
        'loss': loss,
        'perplexity': perplexity
    }


def compare_all_models():
    """Run comparison of all model variants."""
    print("\n" + "#" * 60)
    print("# MODEL COMPARISON: FLOPs AND PERPLEXITY")
    print("#" * 60 + "\n")

    results = {}

    # 1. Baseline model
    results['baseline'] = evaluate_baseline_model()

    # 2. LTP model without pruning (should be similar to baseline)
    results['ltp_no_pruning'] = evaluate_ltp_model(use_token_pruning=False)

    # 3. LTP model with pruning enabled
    results['ltp_with_pruning'] = evaluate_ltp_model(use_token_pruning=True)

    # 4. Dynamic TR model without reduction (should be similar to baseline)
    results['dynamic_tr_no_reduction'] = evaluate_dynamic_tr_model(
        use_token_reduction=False)

    # 5. Dynamic TR model with reduction enabled
    results['dynamic_tr_with_reduction'] = evaluate_dynamic_tr_model(
        use_token_reduction=True)

    # Print summary comparison table
    print("\n" + "#" * 60)
    print("# SUMMARY COMPARISON TABLE")
    print("#" * 60)

    baseline_macs = results['baseline']['macs']

    print(f"\n{'Model':<40} {'MACs':<15} {'Params':<12} {'Perplexity':<12} {'MACs vs Baseline':<15}")
    print("-" * 94)

    model_names = {
        'baseline': 'Baseline GPT',
        'ltp_no_pruning': 'LTP (no pruning)',
        'ltp_with_pruning': 'LTP (with pruning)',
        'dynamic_tr_no_reduction': 'Dynamic TR (no reduction)',
        'dynamic_tr_with_reduction': 'Dynamic TR (with reduction)'
    }

    for key, name in model_names.items():
        r = results[key]
        macs_ratio = r['macs'] / baseline_macs * 100
        print(
            f"{name:<40} {r['macs']:<15.2e} {r['params']:<12.2e} {r['perplexity']:<12.2f} {macs_ratio:>6.1f}%")

    print("\n" + "#" * 60)
    print("# COMPARISON COMPLETE")
    print("#" * 60)

    return results


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Run comparison
    results = compare_all_models()
