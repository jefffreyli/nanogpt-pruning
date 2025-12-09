"""
Perplexity Visualization Script

This script analyzes and visualizes perplexity metrics between:
1. Baseline GPT model
2. LTP (Learned Token Pruning) model
3. Dynamic Token Reduction model
4. Hybrid model (combining Dynamic TR and LTP)

python experiment/visualizations/perplexity_visuals.py
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import Dict, List, Optional, Tuple

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment.models.ltp_model import GPTLTP, GPTConfigLTP
from experiment.models.dynamic_token_reduction_model import GPT as DynamicTRGPT, GPTConfig as DynamicTRGPTConfig
from experiment.models.hybrid_model import GPTHybrid, GPTConfigHybrid
from model import GPT as BaselineGPT, GPTConfig as BaselineGPTConfig


def load_wikitext2_data(data_dir: str = "data/wikitext2"):
    """
    Load WikiText-2 validation data.

    Args:
        data_dir: Directory containing WikiText-2 data files

    Returns:
        Tuple of (val_data, vocab_size)
    """
    project_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    full_data_dir = os.path.join(project_root, data_dir)

    val_path = os.path.join(full_data_dir, "val.bin")
    if not os.path.exists(val_path):
        raise FileNotFoundError(
            f"WikiText-2 validation data not found at {val_path}")

    val_data = np.memmap(val_path, dtype=np.uint16, mode='r')

    # Load vocab size from meta.pkl if available
    meta_path = os.path.join(full_data_dir, "meta.pkl")
    vocab_size = 50304  # default
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        vocab_size = meta.get('vocab_size', 50304)

    return val_data, vocab_size


def evaluate_perplexity(
    model: torch.nn.Module,
    val_data: np.ndarray,
    num_batches: int = 50,
    batch_size: int = 8,
    block_size: int = 512,
    device: str = "cpu",
) -> Tuple[float, float]:
    """
    Evaluate perplexity on WikiText-2 validation data.

    Args:
        model: The model to evaluate
        val_data: WikiText-2 validation data (numpy memmap)
        num_batches: Number of batches to evaluate on
        batch_size: Batch size for evaluation
        block_size: Sequence length
        device: Device to run evaluation on

    Returns:
        Tuple of (average_loss, average_perplexity)
    """
    model.eval()
    model.to(device)

    total_loss = 0.0

    with torch.no_grad():
        for _ in range(num_batches):
            # Sample random positions from validation data
            ix = torch.randint(len(val_data) - block_size, (batch_size,))

            # Load actual WikiText-2 sequences
            x = torch.stack(
                [torch.from_numpy(val_data[i:i+block_size].astype(np.int64)) for i in ix])
            y = torch.stack(
                [torch.from_numpy(val_data[i+1:i+1+block_size].astype(np.int64)) for i in ix])

            x = x.to(device)
            y = y.to(device)

            # Forward pass
            result = model(x, targets=y)

            # Handle different return signatures
            if isinstance(result, tuple):
                logits = result[0]
                loss = result[1]
            else:
                raise ValueError("Model output format not recognized")

            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    avg_perplexity = np.exp(avg_loss)

    return avg_loss, avg_perplexity


def evaluate_ltp_model(
    checkpoint_path: str,
    val_data: np.ndarray,
    num_batches: int = 50,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate LTP model perplexity on WikiText-2 validation data.

    Args:
        checkpoint_path: Path to model checkpoint
        val_data: WikiText-2 validation data
        num_batches: Number of batches for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary with loss and perplexity metrics
    """
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return {
            "loss": float('nan'),
            "perplexity": float('nan'),
            "model_name": "LTP (Learned Token Pruning)",
        }

    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint.get('model_args', {})

    # Filter to only include LTP-specific parameters
    ltp_keys = {'n_layer', 'n_head', 'n_embd', 'block_size', 'bias',
                'vocab_size', 'dropout', 'prune_mode', 'final_token_threshold',
                'temperature', 'masking_mode', 'lambda_factor', 'min_keep_tokens'}
    filtered_args = {k: v for k, v in model_args.items() if k in ltp_keys}

    # Create config from checkpoint
    config = GPTConfigLTP(**filtered_args)
    model = GPTLTP(config)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded LTP checkpoint from {checkpoint_path}")

    loss, perplexity = evaluate_perplexity(
        model,
        val_data,
        num_batches=num_batches,
        batch_size=8,
        block_size=config.block_size,
        device=device,
    )

    return {
        "loss": loss,
        "perplexity": perplexity,
        "model_name": "LTP (Learned Token Pruning)",
    }


def evaluate_baseline_model(
    checkpoint_path: str,
    val_data: np.ndarray,
    num_batches: int = 50,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate baseline GPT model perplexity on WikiText-2 validation data.

    Args:
        checkpoint_path: Path to model checkpoint
        val_data: WikiText-2 validation data
        num_batches: Number of batches for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary with loss and perplexity metrics
    """
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return {
            "loss": float('nan'),
            "perplexity": float('nan'),
            "model_name": "Baseline GPT",
        }

    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint.get('model_args', {})

    # Filter out Dynamic TR-specific parameters that BaselineGPT doesn't accept
    baseline_keys = {'n_layer', 'n_head', 'n_embd', 'block_size', 'bias',
                     'vocab_size', 'dropout'}
    filtered_args = {k: v for k, v in model_args.items() if k in baseline_keys}

    # Create config from checkpoint
    config = BaselineGPTConfig(**filtered_args)
    model = BaselineGPT(config)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded baseline checkpoint from {checkpoint_path}")

    loss, perplexity = evaluate_perplexity(
        model,
        val_data,
        num_batches=num_batches,
        batch_size=8,
        block_size=config.block_size,
        device=device,
    )

    return {
        "loss": loss,
        "perplexity": perplexity,
        "model_name": "Baseline GPT",
    }


def evaluate_dynamic_tr_model(
    checkpoint_path: str,
    val_data: np.ndarray,
    num_batches: int = 50,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate Dynamic Token Reduction model perplexity on WikiText-2 validation data.

    Args:
        checkpoint_path: Path to model checkpoint
        val_data: WikiText-2 validation data
        num_batches: Number of batches for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary with loss and perplexity metrics
    """
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return {
            "loss": float('nan'),
            "perplexity": float('nan'),
            "model_name": "Dynamic Token Reduction",
        }

    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint.get('model_args', {})

    # Filter to only include Dynamic TR parameters
    dynamic_tr_keys = {'n_layer', 'n_head', 'n_embd', 'block_size', 'bias',
                       'vocab_size', 'dropout', 'use_token_reduction',
                       'reduction_layers', 'policy_hidden_dim', 'lambda_tokens', 'rl_weight'}
    filtered_args = {k: v for k, v in model_args.items()
                     if k in dynamic_tr_keys}

    # Create config from checkpoint
    config = DynamicTRGPTConfig(**filtered_args)
    model = DynamicTRGPT(config)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded Dynamic TR checkpoint from {checkpoint_path}")

    loss, perplexity = evaluate_perplexity(
        model,
        val_data,
        num_batches=num_batches,
        batch_size=8,
        block_size=config.block_size,
        device=device,
    )

    return {
        "loss": loss,
        "perplexity": perplexity,
        "model_name": "Dynamic Token Reduction",
    }


def evaluate_hybrid_model(
    checkpoint_path: str,
    val_data: np.ndarray,
    num_batches: int = 50,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate Hybrid model perplexity on WikiText-2 validation data.

    Args:
        checkpoint_path: Path to model checkpoint
        val_data: WikiText-2 validation data
        num_batches: Number of batches for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary with loss and perplexity metrics
    """
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return {
            "loss": float('nan'),
            "perplexity": float('nan'),
            "model_name": "Hybrid (Dynamic TR + LTP)",
        }

    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint.get('model_args', {})

    # Filter to only include Hybrid parameters
    hybrid_keys = {'n_layer', 'n_head', 'n_embd', 'block_size', 'bias',
                   'vocab_size', 'dropout', 'reduction_layers', 'policy_hidden_dim',
                   'lambda_tokens', 'rl_weight', 'ltp_layers', 'final_token_threshold',
                   'temperature', 'masking_mode', 'lambda_factor', 'min_keep_tokens'}
    filtered_args = {k: v for k, v in model_args.items() if k in hybrid_keys}

    # Create config from checkpoint
    config = GPTConfigHybrid(**filtered_args)
    model = GPTHybrid(config)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded Hybrid checkpoint from {checkpoint_path}")

    loss, perplexity = evaluate_perplexity(
        model,
        val_data,
        num_batches=num_batches,
        batch_size=8,
        block_size=config.block_size,
        device=device,
    )

    return {
        "loss": loss,
        "perplexity": perplexity,
        "model_name": "Hybrid (Dynamic TR + LTP)",
    }


def compare_models(
    baseline_checkpoint: str,
    ltp_checkpoint: str,
    dynamic_tr_checkpoint: str,
    hybrid_checkpoint: str,
    val_data: np.ndarray,
    num_batches: int = 50,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """
    Compare perplexity between Baseline, LTP, Dynamic Token Reduction, and Hybrid models on WikiText-2.

    Args:
        baseline_checkpoint: Path to baseline model checkpoint
        ltp_checkpoint: Path to LTP model checkpoint
        dynamic_tr_checkpoint: Path to Dynamic TR model checkpoint
        hybrid_checkpoint: Path to Hybrid model checkpoint
        val_data: WikiText-2 validation data
        num_batches: Number of batches to evaluate
        device: Device to run on

    Returns:
        Dictionary with results for all four models
    """
    print("Evaluating Baseline GPT model...")
    baseline_results = evaluate_baseline_model(
        checkpoint_path=baseline_checkpoint,
        val_data=val_data,
        num_batches=num_batches,
        device=device,
    )

    print("\nEvaluating LTP model...")
    ltp_results = evaluate_ltp_model(
        checkpoint_path=ltp_checkpoint,
        val_data=val_data,
        num_batches=num_batches,
        device=device,
    )

    print("\nEvaluating Dynamic Token Reduction model...")
    dynamic_tr_results = evaluate_dynamic_tr_model(
        checkpoint_path=dynamic_tr_checkpoint,
        val_data=val_data,
        num_batches=num_batches,
        device=device,
    )

    print("\nEvaluating Hybrid model...")
    hybrid_results = evaluate_hybrid_model(
        checkpoint_path=hybrid_checkpoint,
        val_data=val_data,
        num_batches=num_batches,
        device=device,
    )

    return {
        "baseline": baseline_results,
        "ltp": ltp_results,
        "dynamic_tr": dynamic_tr_results,
        "hybrid": hybrid_results,
    }


def analyze_perplexity_vs_sequence_length(
    baseline_checkpoint: str,
    ltp_checkpoint: str,
    dynamic_tr_checkpoint: str,
    hybrid_checkpoint: str,
    val_data: np.ndarray,
    sequence_lengths: List[int] = None,
    num_batches: int = 20,
    device: str = "cpu",
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """
    Create line graphs showing how perplexity scales with sequence length
    for trained models evaluated on WikiText-2.

    Args:
        baseline_checkpoint: Path to baseline checkpoint
        ltp_checkpoint: Path to LTP checkpoint
        dynamic_tr_checkpoint: Path to Dynamic TR checkpoint
        hybrid_checkpoint: Path to Hybrid checkpoint
        val_data: WikiText-2 validation data
        sequence_lengths: List of sequence lengths to test
        num_batches: Number of batches for evaluation
        device: Device to run evaluation on
        save_path: Path to save the figure (optional)
        show_plot: Whether to display the plot
    """
    if sequence_lengths is None:
        sequence_lengths = [64, 128, 256, 512]

    baseline_perplexities = []
    ltp_perplexities = []
    dynamic_tr_perplexities = []
    hybrid_perplexities = []

    print("Analyzing perplexity vs sequence length (trained models on WikiText-2)...")

    # Load all models once with their original block_size from checkpoints
    baseline_model = None
    ltp_model = None
    dynamic_tr_model = None
    hybrid_model = None

    # Load Baseline model
    if os.path.exists(baseline_checkpoint):
        print("  Loading Baseline model...")
        baseline_ckpt = torch.load(baseline_checkpoint, map_location=device)
        model_args = baseline_ckpt.get('model_args', {})
        baseline_keys = {'n_layer', 'n_head', 'n_embd', 'block_size',
                         'bias', 'vocab_size', 'dropout'}
        filtered_args = {k: v for k, v in model_args.items()
                         if k in baseline_keys}
        config = BaselineGPTConfig(**filtered_args)
        baseline_model = BaselineGPT(config)
        baseline_model.load_state_dict(baseline_ckpt["model"])
        baseline_model.to(device)
        baseline_model.eval()

    # Load LTP model
    if os.path.exists(ltp_checkpoint):
        print("  Loading LTP model...")
        ltp_ckpt = torch.load(ltp_checkpoint, map_location=device)
        model_args = ltp_ckpt.get('model_args', {})
        ltp_keys = {'n_layer', 'n_head', 'n_embd', 'block_size', 'bias',
                    'vocab_size', 'dropout', 'prune_mode', 'final_token_threshold',
                    'temperature', 'masking_mode', 'lambda_factor', 'min_keep_tokens'}
        filtered_args = {k: v for k, v in model_args.items() if k in ltp_keys}
        config = GPTConfigLTP(**filtered_args)
        ltp_model = GPTLTP(config)
        ltp_model.load_state_dict(ltp_ckpt["model"])
        ltp_model.to(device)
        ltp_model.eval()

    # Load Dynamic TR model
    if os.path.exists(dynamic_tr_checkpoint):
        print("  Loading Dynamic TR model...")
        dynamic_tr_ckpt = torch.load(
            dynamic_tr_checkpoint, map_location=device)
        model_args = dynamic_tr_ckpt.get('model_args', {})
        dynamic_tr_keys = {'n_layer', 'n_head', 'n_embd', 'block_size', 'bias',
                           'vocab_size', 'dropout', 'use_token_reduction',
                           'reduction_layers', 'policy_hidden_dim', 'lambda_tokens',
                           'rl_weight'}
        filtered_args = {k: v for k, v in model_args.items()
                         if k in dynamic_tr_keys}
        config = DynamicTRGPTConfig(**filtered_args)
        dynamic_tr_model = DynamicTRGPT(config)
        dynamic_tr_model.load_state_dict(dynamic_tr_ckpt["model"])
        dynamic_tr_model.to(device)
        dynamic_tr_model.eval()

    # Load Hybrid model
    if os.path.exists(hybrid_checkpoint):
        print("  Loading Hybrid model...")
        hybrid_ckpt = torch.load(hybrid_checkpoint, map_location=device)
        model_args = hybrid_ckpt.get('model_args', {})
        hybrid_keys = {'n_layer', 'n_head', 'n_embd', 'block_size', 'bias',
                       'vocab_size', 'dropout', 'reduction_layers', 'policy_hidden_dim',
                       'lambda_tokens', 'rl_weight', 'ltp_layers', 'final_token_threshold',
                       'temperature', 'masking_mode', 'lambda_factor', 'min_keep_tokens'}
        filtered_args = {k: v for k, v in model_args.items()
                         if k in hybrid_keys}
        config = GPTConfigHybrid(**filtered_args)
        hybrid_model = GPTHybrid(config)
        hybrid_model.load_state_dict(hybrid_ckpt["model"])
        hybrid_model.to(device)
        hybrid_model.eval()

    # Evaluate each model on different sequence lengths
    for seq_len in sequence_lengths:
        print(f"  Evaluating sequence length: {seq_len}")

        # Baseline
        if baseline_model is not None:
            loss, ppl = evaluate_perplexity(
                baseline_model, val_data, num_batches, 8, seq_len, device)
            baseline_perplexities.append(ppl)
        else:
            baseline_perplexities.append(float('nan'))

        # LTP
        if ltp_model is not None:
            loss, ppl = evaluate_perplexity(
                ltp_model, val_data, num_batches, 8, seq_len, device)
            ltp_perplexities.append(ppl)
        else:
            ltp_perplexities.append(float('nan'))

        # Dynamic TR
        if dynamic_tr_model is not None:
            loss, ppl = evaluate_perplexity(
                dynamic_tr_model, val_data, num_batches, 8, seq_len, device)
            dynamic_tr_perplexities.append(ppl)
        else:
            dynamic_tr_perplexities.append(float('nan'))

        # Hybrid
        if hybrid_model is not None:
            loss, ppl = evaluate_perplexity(
                hybrid_model, val_data, num_batches, 8, seq_len, device)
            hybrid_perplexities.append(ppl)
        else:
            hybrid_perplexities.append(float('nan'))

    # Create line plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        sequence_lengths,
        baseline_perplexities,
        marker="o",
        linewidth=2,
        markersize=8,
        label="Baseline",
        color="#1f77b4",
    )
    ax.plot(
        sequence_lengths,
        ltp_perplexities,
        marker="s",
        linewidth=2,
        markersize=8,
        label="LTP",
        color="#2E86AB",
    )
    ax.plot(
        sequence_lengths,
        dynamic_tr_perplexities,
        marker="^",
        linewidth=2,
        markersize=8,
        label="Dynamic TR",
        color="#A23B72",
    )
    ax.plot(
        sequence_lengths,
        hybrid_perplexities,
        marker="D",
        linewidth=2,
        markersize=8,
        label="Hybrid",
        color="#F18F01",
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


def load_training_history(checkpoint_path: str) -> Optional[Dict]:
    """
    Load training history from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dictionary containing training history or None if not found
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'training_history' in checkpoint:
            history = checkpoint['training_history']
            print(f"Loaded training history from {checkpoint_path}")
            print(f"  - Training steps: {len(history['train_losses'])}")
            print(f"  - Validation steps: {len(history['val_losses'])}")
            return history
        else:
            print(f"No training history found in {checkpoint_path}")
            return None
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None


def plot_training_curves(
    checkpoint_paths: Dict[str, str],
    save_dir: Optional[str] = None,
    show_plot: bool = False,
) -> None:
    """
    Plot training curves (loss and perplexity) from checkpoint histories for all four models:
    Baseline GPT, LTP, Dynamic Token Reduction, and Hybrid.

    Args:
        checkpoint_paths: Dictionary mapping model names to checkpoint paths
                         Expected keys: 'baseline', 'ltp', 'dynamic_tr', 'hybrid'
        save_dir: Directory to save plots (optional)
        show_plot: Whether to display the plots
    """
    histories = {}
    model_labels = {
        'baseline': 'Baseline GPT',
        'ltp': 'LTP (Learned Token Pruning)',
        'dynamic_tr': 'Dynamic Token Reduction',
        'hybrid': 'Hybrid (Dynamic TR + LTP)'
    }

    # Load all histories
    for model_name, ckpt_path in checkpoint_paths.items():
        history = load_training_history(ckpt_path)
        if history:
            histories[model_name] = history

    if not histories:
        print("No training histories found. Skipping training curve plots.")
        return

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {
        'baseline': '#1f77b4',
        'ltp': '#2E86AB',
        'dynamic_tr': '#A23B72',
        'hybrid': '#F18F01'
    }

    # Plot 1: Training Loss vs Steps
    ax = axes[0]
    for model_name, history in histories.items():
        if history['train_losses']:
            steps, losses = zip(*history['train_losses'])
            label = model_labels.get(model_name, model_name)
            color = colors.get(model_name, None)
            ax.plot(steps, losses, label=label,
                    linewidth=2, alpha=0.8, color=color)

    ax.set_xlabel("Training Steps", fontsize=12, fontweight="bold")
    ax.set_ylabel("Training Loss", fontsize=12, fontweight="bold")
    ax.set_title("Training Loss vs Steps\n(Baseline, LTP, Dynamic TR, Hybrid)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")

    # Plot 2: Validation Loss vs Steps
    ax = axes[1]
    for model_name, history in histories.items():
        if history['val_losses']:
            steps, losses = zip(*history['val_losses'])
            label = model_labels.get(model_name, model_name)
            color = colors.get(model_name, None)
            ax.plot(steps, losses, label=label, linewidth=2,
                    alpha=0.8, color=color, marker='o', markersize=4)

    ax.set_xlabel("Training Steps", fontsize=12, fontweight="bold")
    ax.set_ylabel("Validation Loss", fontsize=12, fontweight="bold")
    ax.set_title("Validation Loss vs Steps\n(Baseline, LTP, Dynamic TR, Hybrid)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")

    # Plot 3: Perplexity vs Steps
    ax = axes[2]
    for model_name, history in histories.items():
        if history['val_perplexities']:
            steps, ppls = zip(*history['val_perplexities'])
            label = model_labels.get(model_name, model_name)
            color = colors.get(model_name, None)
            ax.plot(steps, ppls, label=label, linewidth=2,
                    alpha=0.8, color=color, marker='o', markersize=4)

    ax.set_xlabel("Training Steps", fontsize=12, fontweight="bold")
    ax.set_ylabel("Perplexity", fontsize=12, fontweight="bold")
    ax.set_title("Perplexity vs Steps\n(Baseline, LTP, Dynamic TR, Hybrid)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()

    if save_dir:
        save_path = os.path.join(save_dir, "training_curves.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nTraining curves saved to {save_path}")

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
        label="LTP",
        color="#2E86AB",
    )
    ax.plot(
        labels,
        dynamic_tr_perplexities,
        marker="s",
        linewidth=2,
        markersize=8,
        label="Dynamic TR",
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

    # Get project root
    project_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))

    # Load WikiText-2 validation data
    print("Loading WikiText-2 validation data...")
    try:
        val_data, vocab_size = load_wikitext2_data()
        print(
            f"Loaded validation data: {len(val_data):,} tokens, vocab_size={vocab_size}\n")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please prepare WikiText-2 data first.")
        return

    # Define checkpoint paths
    baseline_checkpoint = os.path.join(project_root, "out-baseline-wt2", "baseline.pt")
    ltp_checkpoint = os.path.join(project_root, "out-ltp-wt2", "ltp_ckpt.pt")
    dynamic_tr_checkpoint = os.path.join(project_root, "out-dynamic-tr-wt2", "rl_ckpt.pt")
    hybrid_checkpoint = os.path.join(project_root, "out-hybrid-wt2", "hybrid_ckpt.pt")

    # Compare models on WikiText-2 validation data
    print("=" * 60)
    print("PERPLEXITY ANALYSIS: Baseline vs LTP vs Dynamic TR vs Hybrid")
    print("=" * 60)

    results = compare_models(
        baseline_checkpoint=baseline_checkpoint,
        ltp_checkpoint=ltp_checkpoint,
        dynamic_tr_checkpoint=dynamic_tr_checkpoint,
        hybrid_checkpoint=hybrid_checkpoint,
        val_data=val_data,
        num_batches=50,
        device=device
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (WikiText-2 Validation Set)")
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

    # Training curves from checkpoints
    print("\n--- Training Curves from Checkpoints ---")
    checkpoint_paths = {
        'baseline': baseline_checkpoint,
        'ltp': ltp_checkpoint,
        'dynamic_tr': dynamic_tr_checkpoint,
        'hybrid': hybrid_checkpoint,
    }

    plot_training_curves(
        checkpoint_paths=checkpoint_paths,
        save_dir=output_dir,
        show_plot=False
    )

    # Perplexity vs sequence length (trained models)
    print("\n--- Perplexity vs Sequence Length (Trained Models) ---")
    seq_len_path = os.path.join(
        output_dir, "perplexity_vs_sequence_length.png")
    analyze_perplexity_vs_sequence_length(
        baseline_checkpoint=baseline_checkpoint,
        ltp_checkpoint=ltp_checkpoint,
        dynamic_tr_checkpoint=dynamic_tr_checkpoint,
        hybrid_checkpoint=hybrid_checkpoint,
        val_data=val_data,
        sequence_lengths=[64, 128, 256, 512],
        num_batches=20,
        device=device,
        save_path=seq_len_path,
        show_plot=False
    )

    print("\nAnalysis complete!")
    print(f"\nVisualizations saved to:")
    print(f"  - {output_dir}/training_curves.png")
    print(f"  - {output_dir}/perplexity_vs_sequence_length.png")


if __name__ == "__main__":
    main()
