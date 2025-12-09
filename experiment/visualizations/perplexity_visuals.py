"""
Perplexity Visualization Script

This script analyzes and visualizes perplexity metrics between:
1. Baseline GPT model
2. LTP (Learned Token Pruning) model
3. Dynamic Token Reduction model

It creates line graphs showing relationships across different configurations
(e.g., sequence length, model size) to reveal deeper insights.
"""

from model import GPT as BaselineGPT, GPTConfig as BaselineGPTConfig
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment.models.dynamic_token_reduction_model import GPT as DynamicTRGPT, GPTConfig as DynamicTRGPTConfig
from experiment.models.ltp_model import GPTLTP, GPTConfigLTP


def evaluate_perplexity(
    model: torch.nn.Module,
    num_batches: int = 10,
    batch_size: int = 4,
    block_size: int = 256,
    vocab_size: int = 50304,
    device: str = "cpu",
) -> Tuple[float, float]:
    """
    Evaluate perplexity on multiple batches of random data.

    Args:
        model: The model to evaluate
        num_batches: Number of batches to evaluate on
        batch_size: Batch size for evaluation
        block_size: Sequence length
        vocab_size: Vocabulary size
        device: Device to run evaluation on

    Returns:
        Tuple of (average_loss, average_perplexity)
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_perplexity = 0.0

    with torch.no_grad():
        for _ in range(num_batches):
            # Generate random input and target sequences
            idx = torch.randint(
                0, vocab_size, (batch_size, block_size), device=device)
            targets = torch.randint(
                0, vocab_size, (batch_size, block_size), device=device)

            # Forward pass
            result = model(idx, targets=targets)

            # Handle different return signatures
            if isinstance(result, tuple):
                logits = result[0]
                loss = result[1]
            else:
                raise ValueError("Model output format not recognized")

            # Calculate perplexity
            perplexity = torch.exp(loss)

            total_loss += loss.item()
            total_perplexity += perplexity.item()

    avg_loss = total_loss / num_batches
    avg_perplexity = total_perplexity / num_batches

    return avg_loss, avg_perplexity


def evaluate_ltp_model(
    config: Optional[GPTConfigLTP] = None,
    checkpoint_path: Optional[str] = None,
    num_batches: int = 10,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate LTP model perplexity.

    Args:
        config: Model configuration (uses default if None)
        checkpoint_path: Path to model checkpoint (optional)
        num_batches: Number of batches for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary with loss and perplexity metrics
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

    loss, perplexity = evaluate_perplexity(
        model,
        num_batches=num_batches,
        batch_size=4,
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        device=device,
    )

    return {
        "loss": loss,
        "perplexity": perplexity,
        "model_name": "LTP (Learned Token Pruning)",
    }


def evaluate_baseline_model(
    config: Optional[BaselineGPTConfig] = None,
    checkpoint_path: Optional[str] = None,
    num_batches: int = 10,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate baseline GPT model perplexity.

    Args:
        config: Model configuration (uses default if None)
        checkpoint_path: Path to model checkpoint (optional)
        num_batches: Number of batches for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary with loss and perplexity metrics
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

    loss, perplexity = evaluate_perplexity(
        model,
        num_batches=num_batches,
        batch_size=4,
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        device=device,
    )

    return {
        "loss": loss,
        "perplexity": perplexity,
        "model_name": "Baseline GPT",
    }


def evaluate_dynamic_tr_model(
    config: Optional[DynamicTRGPTConfig] = None,
    checkpoint_path: Optional[str] = None,
    num_batches: int = 10,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate Dynamic Token Reduction model perplexity.

    Args:
        config: Model configuration (uses default if None)
        checkpoint_path: Path to model checkpoint (optional)
        num_batches: Number of batches for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary with loss and perplexity metrics
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

    loss, perplexity = evaluate_perplexity(
        model,
        num_batches=num_batches,
        batch_size=4,
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        device=device,
    )

    return {
        "loss": loss,
        "perplexity": perplexity,
        "model_name": "Dynamic Token Reduction",
    }


def compare_models(
    baseline_config: Optional[BaselineGPTConfig] = None,
    ltp_config: Optional[GPTConfigLTP] = None,
    dynamic_tr_config: Optional[DynamicTRGPTConfig] = None,
    baseline_checkpoint: Optional[str] = None,
    ltp_checkpoint: Optional[str] = None,
    dynamic_tr_checkpoint: Optional[str] = None,
    num_batches: int = 10,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """
    Compare perplexity between Baseline, LTP, and Dynamic Token Reduction models.

    Returns:
        Dictionary with results for all three models
    """
    print("Evaluating Baseline GPT model...")
    baseline_results = evaluate_baseline_model(
        config=baseline_config,
        checkpoint_path=baseline_checkpoint,
        num_batches=num_batches,
        device=device,
    )

    print("\nEvaluating LTP model...")
    ltp_results = evaluate_ltp_model(
        config=ltp_config,
        checkpoint_path=ltp_checkpoint,
        num_batches=num_batches,
        device=device,
    )

    print("\nEvaluating Dynamic Token Reduction model...")
    dynamic_tr_results = evaluate_dynamic_tr_model(
        config=dynamic_tr_config,
        checkpoint_path=dynamic_tr_checkpoint,
        num_batches=num_batches,
        device=device,
    )

    return {
        "baseline": baseline_results,
        "ltp": ltp_results,
        "dynamic_tr": dynamic_tr_results,
    }


def analyze_perplexity_vs_sequence_length(
    sequence_lengths: List[int] = None,
    num_batches: int = 5,
    device: str = "cpu",
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """
    Create line graphs showing how perplexity scales with sequence length.

    Args:
        sequence_lengths: List of sequence lengths to test
        num_batches: Number of batches for evaluation
        device: Device to run evaluation on
        save_path: Path to save the figure (optional)
        show_plot: Whether to display the plot
    """
    if sequence_lengths is None:
        sequence_lengths = [64, 128, 256, 512, 1024]

    baseline_perplexities = []
    ltp_perplexities = []
    dynamic_tr_perplexities = []

    print("Analyzing perplexity vs sequence length...")
    for seq_len in sequence_lengths:
        print(f"  Evaluating sequence length: {seq_len}")

        # Baseline
        baseline_config = BaselineGPTConfig(
            block_size=seq_len,
            vocab_size=50304,
            n_layer=6,
            n_head=6,
            n_embd=384,
            dropout=0.0,
            bias=True,
        )
        baseline_result = evaluate_baseline_model(
            config=baseline_config, num_batches=num_batches, device=device
        )
        baseline_perplexities.append(baseline_result["perplexity"])

        # LTP
        ltp_config = GPTConfigLTP(
            block_size=seq_len,
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
            min_keep_tokens=min(64, seq_len // 4),
        )
        ltp_result = evaluate_ltp_model(
            config=ltp_config, num_batches=num_batches, device=device
        )
        ltp_perplexities.append(ltp_result["perplexity"])

        # Dynamic TR
        dynamic_tr_config = DynamicTRGPTConfig(
            block_size=seq_len,
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
        dynamic_tr_result = evaluate_dynamic_tr_model(
            config=dynamic_tr_config, num_batches=num_batches, device=device
        )
        dynamic_tr_perplexities.append(dynamic_tr_result["perplexity"])

    # Create line plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        sequence_lengths,
        baseline_perplexities,
        marker="o",
        linewidth=2,
        markersize=8,
        label="Baseline GPT",
        color="#1f77b4",
    )
    ax.plot(
        sequence_lengths,
        ltp_perplexities,
        marker="s",
        linewidth=2,
        markersize=8,
        label="LTP (Learned Token Pruning)",
        color="#2E86AB",
    )
    ax.plot(
        sequence_lengths,
        dynamic_tr_perplexities,
        marker="^",
        linewidth=2,
        markersize=8,
        label="Dynamic Token Reduction",
        color="#A23B72",
    )

    ax.set_xlabel("Sequence Length", fontsize=12, fontweight="bold")
    ax.set_ylabel("Perplexity", fontsize=12, fontweight="bold")
    ax.set_title("Perplexity vs Sequence Length",
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


def analyze_perplexity_vs_model_size(
    n_layers_list: List[int] = None,
    num_batches: int = 5,
    device: str = "cpu",
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """
    Create line graphs showing how perplexity changes with model size (number of layers).

    Args:
        n_layers_list: List of number of layers to test
        num_batches: Number of batches for evaluation
        device: Device to run evaluation on
        save_path: Path to save the figure (optional)
        show_plot: Whether to display the plot
    """
    if n_layers_list is None:
        n_layers_list = [4, 6, 8, 10, 12]

    baseline_perplexities = []
    ltp_perplexities = []
    dynamic_tr_perplexities = []

    print("Analyzing perplexity vs model size (number of layers)...")
    for n_layer in n_layers_list:
        print(f"  Evaluating {n_layer} layers")

        # Baseline
        baseline_config = BaselineGPTConfig(
            block_size=256,
            vocab_size=50304,
            n_layer=n_layer,
            n_head=6,
            n_embd=384,
            dropout=0.0,
            bias=True,
        )
        baseline_result = evaluate_baseline_model(
            config=baseline_config, num_batches=num_batches, device=device
        )
        baseline_perplexities.append(baseline_result["perplexity"])

        # LTP
        ltp_config = GPTConfigLTP(
            block_size=256,
            vocab_size=50304,
            n_layer=n_layer,
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
        ltp_result = evaluate_ltp_model(
            config=ltp_config, num_batches=num_batches, device=device
        )
        ltp_perplexities.append(ltp_result["perplexity"])

        # Dynamic TR
        reduction_layers = tuple(
            [i for i in range(1, n_layer) if i % (n_layer // 3) == 0][:2]
        )
        if not reduction_layers:
            reduction_layers = (n_layer // 3, 2 * n_layer // 3)

        dynamic_tr_config = DynamicTRGPTConfig(
            block_size=256,
            vocab_size=50304,
            n_layer=n_layer,
            n_head=6,
            n_embd=384,
            dropout=0.0,
            bias=True,
            use_token_reduction=True,
            reduction_layers=reduction_layers,
            policy_hidden_dim=256,
            lambda_tokens=1e-4,
            rl_weight=0.1,
        )
        dynamic_tr_result = evaluate_dynamic_tr_model(
            config=dynamic_tr_config, num_batches=num_batches, device=device
        )
        dynamic_tr_perplexities.append(dynamic_tr_result["perplexity"])

    # Create line plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        n_layers_list,
        baseline_perplexities,
        marker="o",
        linewidth=2,
        markersize=8,
        label="Baseline GPT",
        color="#1f77b4",
    )
    ax.plot(
        n_layers_list,
        ltp_perplexities,
        marker="s",
        linewidth=2,
        markersize=8,
        label="LTP (Learned Token Pruning)",
        color="#2E86AB",
    )
    ax.plot(
        n_layers_list,
        dynamic_tr_perplexities,
        marker="^",
        linewidth=2,
        markersize=8,
        label="Dynamic Token Reduction",
        color="#A23B72",
    )

    ax.set_xlabel("Number of Layers", fontsize=12, fontweight="bold")
    ax.set_ylabel("Perplexity", fontsize=12, fontweight="bold")
    ax.set_title("Perplexity vs Model Size (Number of Layers)",
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


def create_perplexity_line_plot(
    results_list: List[Dict[str, Dict[str, float]]],
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """
    Create a line plot showing perplexity across different configurations or training steps.

    Args:
        results_list: List of result dictionaries from compare_models()
        labels: Labels for each data point (e.g., training steps, configurations)
        save_path: Path to save the figure (optional)
        show_plot: Whether to display the plot
    """
    if labels is None:
        labels = [f"Config {i+1}" for i in range(len(results_list))]

    ltp_perplexities = []
    dynamic_tr_perplexities = []

    for results in results_list:
        ltp_perplexities.append(results["ltp"]["perplexity"])
        dynamic_tr_perplexities.append(results["dynamic_tr"]["perplexity"])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        labels,
        ltp_perplexities,
        marker="o",
        linewidth=2,
        markersize=8,
        label="LTP (Learned Token Pruning)",
        color="#2E86AB",
    )
    ax.plot(
        labels,
        dynamic_tr_perplexities,
        marker="s",
        linewidth=2,
        markersize=8,
        label="Dynamic Token Reduction",
        color="#A23B72",
    )

    ax.set_xlabel("Configuration / Training Step",
                  fontsize=12, fontweight="bold")
    ax.set_ylabel("Perplexity", fontsize=12, fontweight="bold")
    ax.set_title("Perplexity Comparison Over Configurations",
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


def main():
    """Main function to run perplexity analysis and visualization."""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Compare models with default configurations
    print("=" * 60)
    print("PERPLEXITY ANALYSIS: Baseline vs LTP vs Dynamic Token Reduction")
    print("=" * 60)

    results = compare_models(num_batches=10, device=device)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for model_key, model_results in results.items():
        print(f"\n{model_results['model_name']}:")
        print(f"  Loss: {model_results['loss']:.4f}")
        print(f"  Perplexity: {model_results['perplexity']:.2f}")

    # Create visualizations
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    output_dir = os.path.dirname(os.path.abspath(__file__))

    # Perplexity vs sequence length
    seq_len_path = os.path.join(
        output_dir, "perplexity_vs_sequence_length.png")
    analyze_perplexity_vs_sequence_length(
        num_batches=5, device=device, save_path=seq_len_path, show_plot=False
    )

    # Perplexity vs model size
    model_size_path = os.path.join(output_dir, "perplexity_vs_model_size.png")
    analyze_perplexity_vs_model_size(
        num_batches=5, device=device, save_path=model_size_path, show_plot=False
    )

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
