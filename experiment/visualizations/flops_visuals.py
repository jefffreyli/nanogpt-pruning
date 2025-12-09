"""
FLOPs Visualization Script

python experiment/visualizations/flops_visuals.py
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment.models.hybrid_model import GPTHybrid, GPTConfigHybrid
from experiment.models.ltp_model import GPTLTP, GPTConfigLTP
from experiment.models.dynamic_token_reduction_model import GPT as DynamicTRGPT, GPTConfig as DynamicTRGPTConfig
from model import GPT as BaselineGPT, GPTConfig as BaselineGPTConfig


try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    print("Warning: thop library not available. Install with: pip install thop")
    THOP_AVAILABLE = False


def evaluate_flops(
    model: torch.nn.Module,
    batch_size: int = 1,
    block_size: int = 256,
    vocab_size: int = 50304,
    device: str = "cpu",
) -> Tuple[int, int]:
    """
    Evaluate FLOPs (MACs) and parameters for a model.

    Args:
        model: The model to evaluate
        batch_size: Batch size for evaluation
        block_size: Sequence length
        vocab_size: Vocabulary size
        device: Device to run evaluation on

    Returns:
        Tuple of (macs, params) where:
        - macs: Multiply-Accumulate operations (FLOPs)
        - params: Number of parameters
    """
    if not THOP_AVAILABLE:
        raise ImportError("thop library is required for FLOPs measurement")

    model.eval()
    model.to(device)

    # Create dummy input to run through entire model's forward pass
    dummy_input = torch.randint(
        0, vocab_size, (batch_size, block_size), device=device
    )

    # Calculate FLOPs and parameters using thop
    with torch.no_grad():
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)

    return int(macs), int(params)


def evaluate_ltp_model_flops(
    config: Optional[GPTConfigLTP] = None,
    checkpoint_path: Optional[str] = None,
    batch_size: int = 1,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate LTP model FLOPs.

    Args:
        config: Model configuration (uses default if None)
        checkpoint_path: Path to model checkpoint (optional)
        batch_size: Batch size for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary with FLOPs and parameter metrics
    """
    if config is None:
        config = GPTConfigLTP(
            block_size=256,
            vocab_size=50304,
            n_layer=6,
            n_head=6,
            n_embd=384,
            dropout=0.0,
            bias=True,
            prune_mode="learned",
            final_token_threshold=0.01,
            temperature=5.0,
            masking_mode="hard",
            lambda_factor=0.1,
            min_keep_tokens=64,
        )

    model = GPTLTP(config)

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded checkpoint from {checkpoint_path}")

    macs, params = evaluate_flops(
        model,
        batch_size=batch_size,
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        device=device,
    )

    return {
        "macs": macs,
        "params": params,
        "macs_g": macs / 1e9,  # Giga MACs
        "params_m": params / 1e6,  # Million parameters
        "model_name": "LTP (Learned Token Pruning)",
    }


def evaluate_baseline_model_flops(
    config: Optional[BaselineGPTConfig] = None,
    checkpoint_path: Optional[str] = None,
    batch_size: int = 1,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate baseline GPT model FLOPs.

    Args:
        config: Model configuration (uses default if None)
        checkpoint_path: Path to model checkpoint (optional)
        batch_size: Batch size for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary with FLOPs and parameter metrics
    """
    if config is None:
        config = BaselineGPTConfig(
            block_size=256,
            vocab_size=50304,
            n_layer=6,
            n_head=6,
            n_embd=384,
            dropout=0.0,
            bias=True,
        )

    model = BaselineGPT(config)

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded checkpoint from {checkpoint_path}")

    macs, params = evaluate_flops(
        model,
        batch_size=batch_size,
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        device=device,
    )

    return {
        "macs": macs,
        "params": params,
        "macs_g": macs / 1e9,  # Giga MACs
        "params_m": params / 1e6,  # Million parameters
        "model_name": "Baseline GPT",
    }


def evaluate_dynamic_tr_model_flops(
    config: Optional[DynamicTRGPTConfig] = None,
    checkpoint_path: Optional[str] = None,
    batch_size: int = 1,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate Dynamic Token Reduction model FLOPs.

    Args:
        config: Model configuration (uses default if None)
        checkpoint_path: Path to model checkpoint (optional)
        batch_size: Batch size for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary with FLOPs and parameter metrics
    """
    if config is None:
        config = DynamicTRGPTConfig(
            block_size=256,
            vocab_size=50304,
            n_layer=6,
            n_head=6,
            n_embd=384,
            dropout=0.0,
            bias=True,
            use_token_reduction=True,
            reduction_layers=(2, 4),
            policy_hidden_dim=256,
            lambda_tokens=1e-4,
            rl_weight=0.1,
        )

    model = DynamicTRGPT(config)

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded checkpoint from {checkpoint_path}")

    macs, params = evaluate_flops(
        model,
        batch_size=batch_size,
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        device=device,
    )

    return {
        "macs": macs,
        "params": params,
        "macs_g": macs / 1e9,  # Giga MACs
        "params_m": params / 1e6,  # Million parameters
        "model_name": "Dynamic Token Reduction",
    }


def evaluate_hybrid_model_flops(
    config: Optional[GPTConfigHybrid] = None,
    checkpoint_path: Optional[str] = None,
    batch_size: int = 1,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate Hybrid model FLOPs.

    Args:
        config: Model configuration (uses default if None)
        checkpoint_path: Path to model checkpoint (optional)
        batch_size: Batch size for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary with FLOPs and parameter metrics
    """
    if config is None:
        config = GPTConfigHybrid(
            block_size=256,
            vocab_size=50304,
            n_layer=6,
            n_head=6,
            n_embd=384,
            dropout=0.0,
            bias=True,
            reduction_layers=(2,),
            policy_hidden_dim=256,
            lambda_tokens=1e-4,
            rl_weight=0.1,
            ltp_layers=(3, 4, 5),
            final_token_threshold=0.01,
            temperature=5.0,
            masking_mode="soft",
            lambda_factor=0.1,
            min_keep_tokens=64,
        )

    model = GPTHybrid(config)

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded checkpoint from {checkpoint_path}")

    macs, params = evaluate_flops(
        model,
        batch_size=batch_size,
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        device=device,
    )

    return {
        "macs": macs,
        "params": params,
        "macs_g": macs / 1e9,  # Giga MACs
        "params_m": params / 1e6,  # Million parameters
        "model_name": "Hybrid (Dynamic TR + LTP)",
    }


def compute_layer_flops(seq_len: int, n_embd: int, vocab_size: int) -> Dict[str, int]:
    """
    Compute theoretical FLOPs for a single transformer layer.

    Args:
        seq_len: Sequence length (number of tokens)
        n_embd: Embedding dimension
        vocab_size: Vocabulary size

    Returns:
        Dictionary with FLOPs breakdown per component
    """
    d = n_embd
    T = seq_len

    # Attention: QKV projection (3 linear layers: T * d * d each)
    attn_qkv = 3 * T * d * d * 2  # *2 for multiply-add

    # Attention scores: Q @ K^T -> (T, T), then softmax @ V
    attn_scores = 2 * T * T * d * 2  # Q@K^T and attn@V

    # Attention output projection
    attn_proj = T * d * d * 2

    # MLP: fc1 (d -> 4d) + fc2 (4d -> d)
    mlp = T * d * 4 * d * 2 + T * 4 * d * d * 2  # = 16 * T * d^2

    return {
        "attn_qkv": attn_qkv,
        "attn_scores": attn_scores,
        "attn_proj": attn_proj,
        "mlp": mlp,
        "total": attn_qkv + attn_scores + attn_proj + mlp,
    }


def compute_effective_flops_ltp(
    config: GPTConfigLTP,
    batch_size: int = 1,
    device: str = "cpu",
    checkpoint_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute effective FLOPs for LTP model accounting for token pruning.

    Uses model.get_pruning_stats() to get actual keep ratios per layer,
    then calculates FLOPs scaled by the fraction of active tokens.

    Args:
        config: LTP model configuration
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        checkpoint_path: Optional path to model checkpoint

    Returns:
        Dictionary with effective FLOPs metrics
    """
    model = GPTLTP(config)
    model.eval()
    model.to(device)

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Loading LTP checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])

        # If checkpoint has model_args, we could use them to verify config matches
        if "model_args" in checkpoint:
            print(f"  Checkpoint was trained with: {checkpoint['model_args']}")

    # Create dummy input and get pruning statistics
    dummy_input = torch.randint(
        0, config.vocab_size, (batch_size, config.block_size), device=device
    )

    with torch.no_grad():
        stats = model.get_pruning_stats(dummy_input)

    # Calculate effective FLOPs based on keep ratios
    T = config.block_size
    d = config.n_embd

    # Embedding lookup - THOP typically doesn't count this as MACs
    embedding_flops = 0

    # Calculate per-layer effective FLOPs
    total_effective_flops = embedding_flops

    for layer_idx in range(config.n_layer):
        # Get keep ratio for this layer (default to 1.0 if no stats)
        if stats and layer_idx < len(stats):
            keep_ratio = stats[layer_idx]["keep_ratio"]
        else:
            keep_ratio = 1.0

        # Effective sequence length after pruning
        T_eff = T * keep_ratio

        # LayerNorm 1 (before attention): mean, var, normalize, scale/shift
        # Simplified: ~2 * T_eff * d MACs
        ln1_flops = 2 * T_eff * d

        # Attention QKV projection: 3 linear layers, each is T_eff * d * d MACs
        attn_qkv = 3 * T_eff * d * d

        # Attention scores: Q@K^T (T_eff^2 * d) + attn@V (T_eff^2 * d)
        attn_scores = 2 * T_eff * T_eff * d

        # Attention output projection
        attn_proj = T_eff * d * d

        # LayerNorm 2 (before MLP)
        ln2_flops = 2 * T_eff * d

        # MLP: fc1 (T_eff * d * 4d) + fc2 (T_eff * 4d * d)
        mlp = T_eff * d * 4 * d + T_eff * 4 * d * d  # = 8 * T_eff * d^2

        layer_flops = ln1_flops + attn_qkv + attn_scores + attn_proj + ln2_flops + mlp
        total_effective_flops += layer_flops

    # Final layer norm (use final keep ratio from last layer)
    final_keep_ratio = stats[-1]["keep_ratio"] if stats else 1.0
    final_T = T * final_keep_ratio
    final_ln = 2 * final_T * d

    # LM head: output projection to vocab
    lm_head = final_T * d * config.vocab_size
    total_effective_flops += final_ln + lm_head

    # Get parameter count
    params = sum(p.numel() for p in model.parameters())

    # Calculate average keep ratio for reporting
    avg_keep_ratio = np.mean([s["keep_ratio"]
                             for s in stats]) if stats else 1.0

    return {
        "macs": int(total_effective_flops),
        "params": params,
        "macs_g": total_effective_flops / 1e9,
        "params_m": params / 1e6,
        "model_name": "LTP (Learned Token Pruning)",
        "avg_keep_ratio": avg_keep_ratio,
        "layer_stats": stats,
    }


def compute_effective_flops_dynamic_tr(
    config: DynamicTRGPTConfig,
    batch_size: int = 1,
    device: str = "cpu",
    checkpoint_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute effective FLOPs for Dynamic Token Reduction model.

    Uses model.get_pruning_stats() to get actual keep ratios per layer,
    then calculates FLOPs scaled by the fraction of active tokens.

    Args:
        config: Dynamic TR model configuration
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        checkpoint_path: Optional path to model checkpoint

    Returns:
        Dictionary with effective FLOPs metrics
    """
    model = DynamicTRGPT(config)
    model.eval()
    model.to(device)

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Loading Dynamic TR checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])

        # If checkpoint has model_args, we could use them to verify config matches
        if "model_args" in checkpoint:
            print(f"  Checkpoint was trained with: {checkpoint['model_args']}")

    # Create dummy input and get pruning statistics
    dummy_input = torch.randint(
        0, config.vocab_size, (batch_size, config.block_size), device=device
    )

    with torch.no_grad():
        stats = model.get_pruning_stats(dummy_input)

    T = config.block_size
    d = config.n_embd
    reduction_layers = list(
        config.reduction_layers) if config.use_token_reduction else []

    # Embedding - THOP typically doesn't count this as MACs
    embedding_flops = 0
    total_effective_flops = embedding_flops

    for layer_idx in range(config.n_layer):
        # Get keep ratio for this layer from stats
        if stats and layer_idx < len(stats):
            keep_ratio = stats[layer_idx]["keep_ratio"]
        else:
            keep_ratio = 1.0

        # Effective sequence length after pruning
        T_eff = T * keep_ratio

        # LayerNorm 1 (before attention)
        ln1_flops = 2 * T_eff * d

        # Attention
        attn_qkv = 3 * T_eff * d * d
        attn_scores = 2 * T_eff * T_eff * d
        attn_proj = T_eff * d * d

        # LayerNorm 2 (before MLP)
        ln2_flops = 2 * T_eff * d

        # MLP: fc1 (T_eff * d * 4d) + fc2 (T_eff * 4d * d)
        mlp = T_eff * d * 4 * d + T_eff * 4 * d * d  # = 8 * T_eff * d^2

        layer_flops = ln1_flops + attn_qkv + attn_scores + attn_proj + ln2_flops + mlp

        # Add policy network FLOPs if this is a reduction layer
        if layer_idx in reduction_layers:
            policy_hidden = config.policy_hidden_dim
            # Policy: Linear1 (T_eff * d * hidden) + Linear2 (T_eff * hidden * 1)
            policy_flops = T_eff * d * policy_hidden + T_eff * policy_hidden * 1
            layer_flops += policy_flops

        total_effective_flops += layer_flops

    # Final layer norm and LM head (use final keep ratio)
    final_keep_ratio = stats[-1]["keep_ratio"] if stats else 1.0
    final_T = T * final_keep_ratio
    final_ln = 2 * final_T * d
    lm_head = final_T * d * config.vocab_size
    total_effective_flops += final_ln + lm_head

    # Get parameter count
    params = sum(p.numel() for p in model.parameters())

    # Calculate average keep ratio for reporting
    avg_keep_ratio = np.mean([s["keep_ratio"]
                             for s in stats]) if stats else 1.0

    return {
        "macs": int(total_effective_flops),
        "params": params,
        "macs_g": total_effective_flops / 1e9,
        "params_m": params / 1e6,
        "model_name": "Dynamic Token Reduction",
        "avg_keep_ratio": avg_keep_ratio,
        "reduction_layers": reduction_layers,
        "layer_stats": stats,
    }


def compute_effective_flops_hybrid(
    config: GPTConfigHybrid,
    batch_size: int = 1,
    device: str = "cpu",
    checkpoint_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute effective FLOPs for Hybrid model accounting for both token reduction and pruning.

    Uses model.get_pruning_stats() to get actual keep ratios per layer,
    then calculates FLOPs scaled by the fraction of active tokens.

    Args:
        config: Hybrid model configuration
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        checkpoint_path: Optional path to model checkpoint

    Returns:
        Dictionary with effective FLOPs metrics
    """
    model = GPTHybrid(config)
    model.eval()
    model.to(device)

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Loading Hybrid checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])

        # If checkpoint has model_args, we could use them to verify config matches
        if "model_args" in checkpoint:
            print(f"  Checkpoint was trained with: {checkpoint['model_args']}")

    # Create dummy input and get pruning statistics
    dummy_input = torch.randint(
        0, config.vocab_size, (batch_size, config.block_size), device=device
    )

    with torch.no_grad():
        stats = model.get_pruning_stats(dummy_input)

    T = config.block_size
    d = config.n_embd

    # Embedding - THOP typically doesn't count this as MACs
    embedding_flops = 0
    total_effective_flops = embedding_flops

    for layer_idx in range(config.n_layer):
        # Get keep ratio for this layer from stats
        if stats and layer_idx < len(stats):
            keep_ratio = stats[layer_idx]["keep_ratio"]
        else:
            keep_ratio = 1.0

        # Effective sequence length after pruning
        T_eff = T * keep_ratio

        # LayerNorm 1 (before attention)
        ln1_flops = 2 * T_eff * d

        # Attention
        attn_qkv = 3 * T_eff * d * d
        attn_scores = 2 * T_eff * T_eff * d
        attn_proj = T_eff * d * d

        # LayerNorm 2 (before MLP)
        ln2_flops = 2 * T_eff * d

        # MLP: fc1 (T_eff * d * 4d) + fc2 (T_eff * 4d * d)
        mlp = T_eff * d * 4 * d + T_eff * 4 * d * d  # = 8 * T_eff * d^2

        layer_flops = ln1_flops + attn_qkv + attn_scores + attn_proj + ln2_flops + mlp

        # Add policy network FLOPs if this is a reduction layer
        if layer_idx in config.reduction_layers:
            policy_hidden = config.policy_hidden_dim
            # Policy: Linear1 (T_eff * d * hidden) + Linear2 (T_eff * hidden * 1)
            policy_flops = T_eff * d * policy_hidden + T_eff * policy_hidden * 1
            layer_flops += policy_flops

        total_effective_flops += layer_flops

    # Final layer norm and LM head (use final keep ratio)
    final_keep_ratio = stats[-1]["keep_ratio"] if stats else 1.0
    final_T = T * final_keep_ratio
    final_ln = 2 * final_T * d
    lm_head = final_T * d * config.vocab_size
    total_effective_flops += final_ln + lm_head

    # Get parameter count
    params = sum(p.numel() for p in model.parameters())

    # Calculate average keep ratio for reporting
    avg_keep_ratio = np.mean([s["keep_ratio"]
                             for s in stats]) if stats else 1.0

    return {
        "macs": int(total_effective_flops),
        "params": params,
        "macs_g": total_effective_flops / 1e9,
        "params_m": params / 1e6,
        "model_name": "Hybrid (Dynamic TR + LTP)",
        "avg_keep_ratio": avg_keep_ratio,
        "layer_stats": stats,
    }


def compare_models_flops(
    baseline_config: Optional[BaselineGPTConfig] = None,
    ltp_config: Optional[GPTConfigLTP] = None,
    dynamic_tr_config: Optional[DynamicTRGPTConfig] = None,
    hybrid_config: Optional[GPTConfigHybrid] = None,
    baseline_checkpoint: Optional[str] = None,
    ltp_checkpoint: Optional[str] = None,
    dynamic_tr_checkpoint: Optional[str] = None,
    hybrid_checkpoint: Optional[str] = None,
    batch_size: int = 1,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """
    Compare FLOPs between Baseline, LTP, Dynamic Token Reduction, and Hybrid models.

    Returns:
        Dictionary with results for all four models
    """
    print("Evaluating Baseline GPT model FLOPs...")
    baseline_results = evaluate_baseline_model_flops(
        config=baseline_config,
        checkpoint_path=baseline_checkpoint,
        batch_size=batch_size,
        device=device,
    )

    print("\nEvaluating LTP model FLOPs...")
    ltp_results = evaluate_ltp_model_flops(
        config=ltp_config,
        checkpoint_path=ltp_checkpoint,
        batch_size=batch_size,
        device=device,
    )

    print("\nEvaluating Dynamic Token Reduction model FLOPs...")
    dynamic_tr_results = evaluate_dynamic_tr_model_flops(
        config=dynamic_tr_config,
        checkpoint_path=dynamic_tr_checkpoint,
        batch_size=batch_size,
        device=device,
    )

    print("\nEvaluating Hybrid model FLOPs...")
    hybrid_results = evaluate_hybrid_model_flops(
        config=hybrid_config,
        checkpoint_path=hybrid_checkpoint,
        batch_size=batch_size,
        device=device,
    )

    return {
        "baseline": baseline_results,
        "ltp": ltp_results,
        "dynamic_tr": dynamic_tr_results,
        "hybrid": hybrid_results,
    }


def analyze_flops_vs_sequence_length(
    sequence_lengths: List[int] = None,
    device: str = "cpu",
    save_path: Optional[str] = None,
    show_plot: bool = True,
    baseline_checkpoint: Optional[str] = None,
    ltp_checkpoint: Optional[str] = None,
    dynamic_tr_checkpoint: Optional[str] = None,
    hybrid_checkpoint: Optional[str] = None,
) -> None:
    """
    Create line graphs showing how FLOPs scale with sequence length.
    As input sequences get longer, how much compute do we actually save with token pruning compared to baseline GPT?

    Args:
        sequence_lengths: List of sequence lengths to test
        device: Device to run evaluation on
        save_path: Path to save the figure (optional)
        show_plot: Whether to display the plot
        baseline_checkpoint: Path to trained baseline model checkpoint (optional)
        ltp_checkpoint: Path to trained LTP model checkpoint (optional)
        dynamic_tr_checkpoint: Path to trained Dynamic TR model checkpoint (optional)
        hybrid_checkpoint: Path to trained Hybrid model checkpoint (optional)
    """
    if sequence_lengths is None:
        sequence_lengths = [64, 128, 256, 512, 1024]

    baseline_flops = []
    ltp_flops = []
    dynamic_tr_flops = []
    hybrid_flops = []

    print("Analyzing FLOPs vs sequence length...")
    for seq_len in sequence_lengths:
        print(f"  Evaluating sequence length: {seq_len}")

        # Baseline - GPT-2 architecture
        baseline_config = BaselineGPTConfig(
            block_size=seq_len,
            vocab_size=50257,  # GPT-2 vocab size
            n_layer=12,
            n_head=12,
            n_embd=768,
            dropout=0.0,
            bias=True,
        )
        # Only load checkpoint if sequence length matches checkpoint's block_size (512)
        baseline_ckpt = baseline_checkpoint if seq_len == 512 else None
        baseline_result = evaluate_baseline_model_flops(
            config=baseline_config,
            checkpoint_path=baseline_ckpt,
            device=device
        )
        baseline_flops.append(baseline_result["macs_g"])

        # LTP - use effective FLOPs that account for token pruning (GPT-2 architecture)
        ltp_config = GPTConfigLTP(
            block_size=seq_len,
            vocab_size=50257,  # GPT-2 vocab size
            n_layer=12,
            n_head=12,
            n_embd=768,
            dropout=0.0,
            bias=True,
            prune_mode="learned",
            final_token_threshold=0.01,
            temperature=5.0,
            masking_mode="hard",
            lambda_factor=0.1,
            min_keep_tokens=min(128, seq_len // 4),
        )
        # Only load checkpoint if sequence length matches checkpoint's block_size (512)
        ltp_ckpt = ltp_checkpoint if seq_len == 512 else None
        ltp_result = compute_effective_flops_ltp(
            config=ltp_config, device=device, checkpoint_path=ltp_ckpt)
        ltp_flops.append(ltp_result["macs_g"])

        # Dynamic TR - use effective FLOPs that account for token reduction (GPT-2 architecture)
        dynamic_tr_config = DynamicTRGPTConfig(
            block_size=seq_len,
            vocab_size=50257,  # GPT-2 vocab size
            n_layer=12,
            n_head=12,
            n_embd=768,
            dropout=0.0,
            bias=True,
            use_token_reduction=True,
            reduction_layers=(4, 8),  # Layers 4 and 8 for 12-layer model
            policy_hidden_dim=256,
            lambda_tokens=1e-4,
            rl_weight=0.1,
        )
        # Only load checkpoint if sequence length matches checkpoint's block_size (512)
        dynamic_tr_ckpt = dynamic_tr_checkpoint if seq_len == 512 else None
        dynamic_tr_result = compute_effective_flops_dynamic_tr(
            config=dynamic_tr_config, device=device, checkpoint_path=dynamic_tr_ckpt
        )
        dynamic_tr_flops.append(dynamic_tr_result["macs_g"])

        # Hybrid - use effective FLOPs that account for both token reduction and pruning (GPT-2 architecture)
        hybrid_config = GPTConfigHybrid(
            block_size=seq_len,
            vocab_size=50257,  # GPT-2 vocab size
            n_layer=12,
            n_head=12,
            n_embd=768,
            dropout=0.0,
            bias=True,
            reduction_layers=(4, 8),  # Layers 4 and 8 for 12-layer model
            policy_hidden_dim=256,
            lambda_tokens=1e-4,
            rl_weight=0.1,
            ltp_layers=(9, 10, 11),  # Layers 9, 10, 11 for LTP
            final_token_threshold=0.01,
            temperature=5.0,
            masking_mode="soft",
            lambda_factor=0.1,
            min_keep_tokens=min(128, seq_len // 4),
        )
        # Only load checkpoint if sequence length matches checkpoint's block_size (512)
        hybrid_ckpt = hybrid_checkpoint if seq_len == 512 else None
        hybrid_result = compute_effective_flops_hybrid(
            config=hybrid_config, device=device, checkpoint_path=hybrid_ckpt
        )
        hybrid_flops.append(hybrid_result["macs_g"])

    # Create line plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        sequence_lengths,
        baseline_flops,
        marker="o",
        linewidth=2,
        markersize=8,
        label="Baseline",
        color="#1f77b4",
    )
    ax.plot(
        sequence_lengths,
        ltp_flops,
        marker="s",
        linewidth=2,
        markersize=8,
        label="LTP",
        color="#2E86AB",
    )
    ax.plot(
        sequence_lengths,
        dynamic_tr_flops,
        marker="^",
        linewidth=2,
        markersize=8,
        label="Dynamic TR",
        color="#A23B72",
    )
    ax.plot(
        sequence_lengths,
        hybrid_flops,
        marker="D",
        linewidth=2,
        markersize=8,
        label="Hybrid",
        color="#F18F01",
    )

    ax.set_xlabel("Sequence Length", fontsize=12, fontweight="bold")
    ax.set_ylabel("Effective FLOPs (Giga MACs)",
                  fontsize=12, fontweight="bold")
    ax.set_title("Effective FLOPs vs Sequence Length\n(Accounting for Token Pruning)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def analyze_flops_vs_model_size(
    n_layers_list: List[int] = None,
    device: str = "cpu",
    save_path: Optional[str] = None,
    show_plot: bool = True,
    baseline_checkpoint: Optional[str] = None,
    ltp_checkpoint: Optional[str] = None,
    dynamic_tr_checkpoint: Optional[str] = None,
    hybrid_checkpoint: Optional[str] = None,
) -> None:
    """
    Create line graphs showing how FLOPs change with model size (number of layers).

    Args:
        n_layers_list: List of number of layers to test
        device: Device to run evaluation on
        save_path: Path to save the figure (optional)
        show_plot: Whether to display the plot
        baseline_checkpoint: Path to trained baseline model checkpoint (optional)
        ltp_checkpoint: Path to trained LTP model checkpoint (optional)
        dynamic_tr_checkpoint: Path to trained Dynamic TR model checkpoint (optional)
        hybrid_checkpoint: Path to trained Hybrid model checkpoint (optional)
    """
    if n_layers_list is None:
        n_layers_list = [4, 6, 8, 10, 12]

    baseline_flops = []
    ltp_flops = []
    dynamic_tr_flops = []
    hybrid_flops = []

    print("Analyzing FLOPs vs model size (number of layers)...")
    for n_layer in n_layers_list:
        print(f"  Evaluating {n_layer} layers")

        # Baseline - GPT-2 architecture
        baseline_config = BaselineGPTConfig(
            block_size=512,  # Match checkpoint block size
            vocab_size=50257,  # GPT-2 vocab size
            n_layer=n_layer,
            n_head=12,
            n_embd=768,
            dropout=0.0,
            bias=True,
        )
        # Only load checkpoint if n_layer matches checkpoint's n_layer (12)
        baseline_ckpt = baseline_checkpoint if n_layer == 12 else None
        baseline_result = evaluate_baseline_model_flops(
            config=baseline_config,
            checkpoint_path=baseline_ckpt,
            device=device
        )
        baseline_flops.append(baseline_result["macs_g"])

        # LTP - use effective FLOPs that account for token pruning (GPT-2 architecture)
        ltp_config = GPTConfigLTP(
            block_size=512,  # Match checkpoint block size
            vocab_size=50257,  # GPT-2 vocab size
            n_layer=n_layer,
            n_head=12,
            n_embd=768,
            dropout=0.0,
            bias=True,
            prune_mode="learned",
            final_token_threshold=0.01,
            temperature=5.0,
            masking_mode="hard",
            lambda_factor=0.1,
            min_keep_tokens=128,
        )
        # Only load checkpoint if n_layer matches checkpoint's n_layer (12)
        ltp_ckpt = ltp_checkpoint if n_layer == 12 else None
        ltp_result = compute_effective_flops_ltp(
            config=ltp_config, device=device, checkpoint_path=ltp_ckpt)
        ltp_flops.append(ltp_result["macs_g"])

        # Dynamic TR - use effective FLOPs that account for token reduction (GPT-2 architecture)
        reduction_layers = tuple(
            [i for i in range(1, n_layer) if i % (n_layer // 3) == 0][:2]
        )
        if not reduction_layers:
            reduction_layers = (n_layer // 3, 2 * n_layer // 3)

        dynamic_tr_config = DynamicTRGPTConfig(
            block_size=512,  # Match checkpoint block size
            vocab_size=50257,  # GPT-2 vocab size
            n_layer=n_layer,
            n_head=12,
            n_embd=768,
            dropout=0.0,
            bias=True,
            use_token_reduction=True,
            reduction_layers=reduction_layers,
            policy_hidden_dim=256,
            lambda_tokens=1e-4,
            rl_weight=0.1,
        )
        # Only load checkpoint if n_layer matches checkpoint's n_layer (12)
        dynamic_tr_ckpt = dynamic_tr_checkpoint if n_layer == 12 else None
        dynamic_tr_result = compute_effective_flops_dynamic_tr(
            config=dynamic_tr_config, device=device, checkpoint_path=dynamic_tr_ckpt
        )
        dynamic_tr_flops.append(dynamic_tr_result["macs_g"])

        # Hybrid - use effective FLOPs that account for both token reduction and pruning (GPT-2 architecture)
        ltp_layers = tuple(range(max(0, n_layer - 3), n_layer))
        hybrid_config = GPTConfigHybrid(
            block_size=512,  # Match checkpoint block size
            vocab_size=50257,  # GPT-2 vocab size
            n_layer=n_layer,
            n_head=12,
            n_embd=768,
            dropout=0.0,
            bias=True,
            reduction_layers=reduction_layers,
            policy_hidden_dim=256,
            lambda_tokens=1e-4,
            rl_weight=0.1,
            ltp_layers=ltp_layers,
            final_token_threshold=0.01,
            temperature=5.0,
            masking_mode="soft",
            lambda_factor=0.1,
            min_keep_tokens=128,
        )
        # Only load checkpoint if n_layer matches checkpoint's n_layer (12)
        hybrid_ckpt = hybrid_checkpoint if n_layer == 12 else None
        hybrid_result = compute_effective_flops_hybrid(
            config=hybrid_config, device=device, checkpoint_path=hybrid_ckpt
        )
        hybrid_flops.append(hybrid_result["macs_g"])

    # Create line plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        n_layers_list,
        baseline_flops,
        marker="o",
        linewidth=2,
        markersize=8,
        label="Baseline",
        color="#1f77b4",
    )
    ax.plot(
        n_layers_list,
        ltp_flops,
        marker="s",
        linewidth=2,
        markersize=8,
        label="LTP",
        color="#2E86AB",
    )
    ax.plot(
        n_layers_list,
        dynamic_tr_flops,
        marker="^",
        linewidth=2,
        markersize=8,
        label="Dynamic TR",
        color="#A23B72",
    )
    ax.plot(
        n_layers_list,
        hybrid_flops,
        marker="D",
        linewidth=2,
        markersize=8,
        label="Hybrid",
        color="#F18F01",
    )

    ax.set_xlabel("Number of Layers", fontsize=12, fontweight="bold")
    ax.set_ylabel("Effective FLOPs (Giga MACs)",
                  fontsize=12, fontweight="bold")
    ax.set_title("Effective FLOPs vs Model Size (Number of Layers)\n(Accounting for Token Pruning)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle="--")
    # ax.set_yscale("log")  # Removed log scale to allow y-axis to start at 0
    ax.set_ylim(bottom=0)  # Start y-axis at 0

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def create_flops_efficiency_graph(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """
    Create a graph showing FLOPs efficiency (FLOPs per parameter).

    Args:
        results: Dictionary with model results from compare_models_flops()
        save_path: Path to save the figure (optional)
        show_plot: Whether to display the plot
    """
    models = []
    flops_per_param = []
    colors = []

    # Extract data and calculate efficiency
    for model_key, model_results in results.items():
        models.append(model_results["model_name"])
        # FLOPs per parameter ratio
        efficiency = (
            model_results["macs"] / model_results["params"]
            if model_results["params"] > 0
            else 0
        )
        flops_per_param.append(efficiency)
        colors.append(
            "#2E86AB" if "LTP" in model_results["model_name"] else "#A23B72"
        )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        models, flops_per_param, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5
    )
    ax.set_ylabel("FLOPs per Parameter", fontsize=12, fontweight="bold")
    ax.set_title("Computational Efficiency (FLOPs per Parameter)",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for bar, eff_val in zip(bars, flops_per_param):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{eff_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Rotate x-axis labels if needed
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def create_flops_line_plot(
    results_list: List[Dict[str, Dict[str, float]]],
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """
    Create a line plot showing FLOPs across different configurations or training steps.

    Args:
        results_list: List of result dictionaries from compare_models_flops()
        labels: Labels for each data point (e.g., training steps, configurations)
        save_path: Path to save the figure (optional)
        show_plot: Whether to display the plot
    """
    if labels is None:
        labels = [f"Config {i+1}" for i in range(len(results_list))]

    ltp_flops = []
    dynamic_tr_flops = []

    for results in results_list:
        ltp_flops.append(results["ltp"]["macs_g"])
        dynamic_tr_flops.append(results["dynamic_tr"]["macs_g"])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        labels,
        ltp_flops,
        marker="o",
        linewidth=2,
        markersize=8,
        label="LTP",
        color="#2E86AB",
    )
    ax.plot(
        labels,
        dynamic_tr_flops,
        marker="s",
        linewidth=2,
        markersize=8,
        label="Dynamic TR",
        color="#A23B72",
    )

    ax.set_xlabel("Configuration / Training Step",
                  fontsize=12, fontweight="bold")
    ax.set_ylabel("FLOPs (Giga MACs)", fontsize=12, fontweight="bold")
    ax.set_title("FLOPs Comparison Over Configurations",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    """Main function to run FLOPs analysis and visualization."""
    if not THOP_AVAILABLE:
        print("Error: thop library is required for FLOPs measurement.")
        print("Install it with: pip install thop")
        return

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Compare models with default configurations (GPT-2 architecture)
    print("=" * 60)
    print("FLOPs ANALYSIS: Baseline vs LTP vs Dynamic TR vs Hybrid")
    print("=" * 60)

    # Use GPT-2 architecture for fair comparison
    baseline_config = BaselineGPTConfig(
        block_size=512,
        vocab_size=50257,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,
    )

    ltp_config = GPTConfigLTP(
        block_size=512,
        vocab_size=50257,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,
        prune_mode="learned",
        final_token_threshold=0.01,
        temperature=5.0,
        masking_mode="hard",
        lambda_factor=0.1,
        min_keep_tokens=128,
    )

    dynamic_tr_config = DynamicTRGPTConfig(
        block_size=512,
        vocab_size=50257,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,
        use_token_reduction=True,
        reduction_layers=(4, 8),
        policy_hidden_dim=256,
        lambda_tokens=1e-4,
        rl_weight=0.1,
    )

    hybrid_config = GPTConfigHybrid(
        block_size=512,
        vocab_size=50257,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,
        reduction_layers=(4, 8),
        policy_hidden_dim=256,
        lambda_tokens=1e-4,
        rl_weight=0.1,
        ltp_layers=(9, 10, 11),
        final_token_threshold=0.01,
        temperature=5.0,
        masking_mode="soft",
        lambda_factor=0.1,
        min_keep_tokens=128,
    )

    results = compare_models_flops(
        baseline_config=baseline_config,
        ltp_config=ltp_config,
        dynamic_tr_config=dynamic_tr_config,
        hybrid_config=hybrid_config,
        batch_size=1,
        device=device
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for model_key, model_results in results.items():
        print(f"\n{model_results['model_name']}:")
        print(f"  FLOPs (MACs): {model_results['macs']:,}")
        print(f"  FLOPs (Giga MACs): {model_results['macs_g']:.2f}G")
        print(f"  Parameters: {model_results['params']:,}")
        print(f"  Parameters (Millions): {model_results['params_m']:.2f}M")
        print(
            f"  FLOPs per Parameter: {model_results['macs'] / model_results['params']:.2f}")

    # Calculate relative comparison
    baseline_macs = results["baseline"]["macs"]
    print("\n" + "=" * 60)
    print("RELATIVE COMPARISON (vs Baseline)")
    print("=" * 60)
    for model_key, model_results in results.items():
        if model_key != "baseline":
            ratio = model_results["macs"] / baseline_macs * 100
            print(
                f"{model_results['model_name']} FLOPs: {ratio:.1f}% of Baseline"
            )

    # Create visualizations
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    output_dir = os.path.dirname(os.path.abspath(__file__))

    # Look for trained model checkpoints
    project_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    # Training saves to out-ltp-wt2/ckpt.pt by default (see train_ltp_wt2.py config)
    baseline_checkpoint = os.path.join(project_root, "out-baseline-wt2", "baseline.pt")
    ltp_checkpoint = os.path.join(project_root, "out-ltp-wt2", "ltp_ckpt.pt")
    dynamic_tr_checkpoint = os.path.join(project_root, "out-dynamic-tr-wt2", "rl_ckpt.pt")
    hybrid_checkpoint = os.path.join(project_root, "out-hybrid-wt2", "hybrid_ckpt.pt")

    # Check if checkpoints exist
    if os.path.exists(baseline_checkpoint):
        print(f"\nFound trained baseline checkpoint: {baseline_checkpoint}")
        print("Using trained model for baseline analysis...")
    else:
        print(
            f"\nNo trained baseline checkpoint found at {baseline_checkpoint}")
        print("Using randomly initialized model for baseline analysis...")
        baseline_checkpoint = None

    # Check if checkpoints exist
    if os.path.exists(ltp_checkpoint):
        print(f"\nFound trained LTP checkpoint: {ltp_checkpoint}")
        print("Using trained model for LTP analysis...")
    else:
        print(f"\nNo trained LTP checkpoint found at {ltp_checkpoint}")
        print("Using randomly initialized model for LTP analysis...")
        ltp_checkpoint = None

    if os.path.exists(dynamic_tr_checkpoint):
        print(
            f"\nFound trained Dynamic TR checkpoint: {dynamic_tr_checkpoint}")
        print("Using trained model for Dynamic TR analysis...")
    else:
        print(
            f"\nNo trained Dynamic TR checkpoint found at {dynamic_tr_checkpoint}")
        print("Using randomly initialized model for Dynamic TR analysis...")
        dynamic_tr_checkpoint = None

    if os.path.exists(hybrid_checkpoint):
        print(f"\nFound trained Hybrid checkpoint: {hybrid_checkpoint}")
        print("Using trained model for Hybrid analysis...")
    else:
        print(f"\nNo trained Hybrid checkpoint found at {hybrid_checkpoint}")
        print("Using randomly initialized model for Hybrid analysis...")
        hybrid_checkpoint = None

    # FLOPs vs sequence length
    seq_len_path = os.path.join(output_dir, "flops_vs_sequence_length.png")
    analyze_flops_vs_sequence_length(
        device=device,
        save_path=seq_len_path,
        show_plot=False,
        baseline_checkpoint=baseline_checkpoint,
        ltp_checkpoint=ltp_checkpoint,
        dynamic_tr_checkpoint=dynamic_tr_checkpoint,
        hybrid_checkpoint=hybrid_checkpoint,
    )

    # FLOPs vs model size
    model_size_path = os.path.join(output_dir, "flops_vs_model_size.png")
    analyze_flops_vs_model_size(
        device=device,
        save_path=model_size_path,
        show_plot=False,
        baseline_checkpoint=baseline_checkpoint,
        ltp_checkpoint=ltp_checkpoint,
        dynamic_tr_checkpoint=dynamic_tr_checkpoint,
        hybrid_checkpoint=hybrid_checkpoint,
    )

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
