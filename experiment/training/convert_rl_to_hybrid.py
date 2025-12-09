"""
Convert a trained RL-based token reduction model to a hybrid RL+LTP architecture.

This script:
1. Loads a checkpoint from a trained RL model (GPT class)
2. Creates a new GPTHybrid model with LTP layers added
3. Transfers weights from RL model to hybrid model
4. Saves the converted checkpoint for further training

Usage:
    python experiment/training/convert_rl_to_hybrid.py \
        --rl_checkpoint out-dynamic-tr-wt2/ckpt.pt \
        --output out-hybrid-rl-ltp/converted_ckpt.pt \
        --ltp_layers 9 10 11
"""

import argparse
import os
import sys
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from experiment.models.dynamic_token_reduction_model import GPT, GPTConfig
from experiment.models.hybrid_model import GPTHybrid, GPTConfigHybrid


def convert_rl_to_hybrid(
    rl_checkpoint_path: str,
    output_path: str,
    ltp_layers: tuple = (9, 10, 11),
    verbose: bool = True,
):
    """
    Convert RL model checkpoint to hybrid architecture.

    Args:
        rl_checkpoint_path: Path to trained RL model checkpoint
        output_path: Path to save converted hybrid checkpoint
        ltp_layers: Which layers to use LTP (default: last 3 layers)
        verbose: Print detailed conversion info
    """

    if verbose:
        print(f"Loading RL checkpoint from: {rl_checkpoint_path}")

    # Load RL checkpoint
    rl_checkpoint = torch.load(rl_checkpoint_path, map_location='cpu')
    rl_model_args = rl_checkpoint['model_args']
    rl_state_dict = rl_checkpoint['model']

    if verbose:
        print(f"RL model config: {rl_model_args}")
        print(f"Number of RL state dict keys: {len(rl_state_dict)}")

    # Create hybrid model config based on RL config
    hybrid_args = {
        'n_layer': rl_model_args.get('n_layer', 12),
        'n_head': rl_model_args.get('n_head', 12),
        'n_embd': rl_model_args.get('n_embd', 768),
        'block_size': rl_model_args.get('block_size', 512),
        'bias': rl_model_args.get('bias', True),
        'vocab_size': rl_model_args.get('vocab_size', 50304),
        'dropout': rl_model_args.get('dropout', 0.0),

        # Dynamic TR params (from RL model)
        'reduction_layers': rl_model_args.get('reduction_layers', (4, 8)),
        'policy_hidden_dim': rl_model_args.get('policy_hidden_dim', 256),
        'lambda_tokens': rl_model_args.get('lambda_tokens', 1e-4),
        'rl_weight': rl_model_args.get('rl_weight', 0.1),

        # LTP params (new)
        'ltp_layers': tuple(ltp_layers),
        'final_token_threshold': 0.01,
        'temperature': 5.0,
        'masking_mode': 'soft',
        'lambda_factor': 0.1,
        'min_keep_tokens': 64,
    }

    if verbose:
        print(f"\nCreating hybrid model with config:")
        print(f"  Reduction layers (RL): {hybrid_args['reduction_layers']}")
        print(f"  LTP layers: {hybrid_args['ltp_layers']}")

    # Create hybrid model
    hybrid_config = GPTConfigHybrid(**hybrid_args)
    hybrid_model = GPTHybrid(hybrid_config)
    hybrid_state_dict = hybrid_model.state_dict()

    if verbose:
        print(
            f"\nHybrid model created with {len(hybrid_state_dict)} parameters")

    # Transfer weights from RL to hybrid
    # We need to handle:
    # 1. Standard transformer blocks (layers not in ltp_layers)
    # 2. RL policies (same in both models)
    # 3. LTP blocks need special handling (convert Block -> BlockLTP)

    transferred_keys = []
    missing_keys = []
    conversion_notes = []

    # Remove '_orig_mod.' prefix if present (from torch.compile)
    unwanted_prefix = '_orig_mod.'
    rl_state_dict_clean = {}
    for k, v in rl_state_dict.items():
        if k.startswith(unwanted_prefix):
            rl_state_dict_clean[k[len(unwanted_prefix):]] = v
        else:
            rl_state_dict_clean[k] = v

    rl_state_dict = rl_state_dict_clean

    # Transfer weights
    for key in hybrid_state_dict.keys():
        # Check if this key exists in RL model
        if key in rl_state_dict:
            # Direct transfer for matching keys
            if rl_state_dict[key].shape == hybrid_state_dict[key].shape:
                hybrid_state_dict[key] = rl_state_dict[key].clone()
                transferred_keys.append(key)
            else:
                conversion_notes.append(
                    f"Shape mismatch for {key}: RL {rl_state_dict[key].shape} vs Hybrid {hybrid_state_dict[key].shape}"
                )
        else:
            # Key doesn't exist in RL model
            # This is expected for LTP-specific parameters in layers 9, 10, 11
            if any(f'transformer.h.{layer}.' in key for layer in ltp_layers):
                # LTP layer parameters - these are newly initialized
                if 'threshold' in key:
                    conversion_notes.append(
                        f"LTP threshold parameter (new): {key}")
                elif 'attn' in key or 'mlp' in key or 'ln' in key:
                    # Try to transfer from RL model's standard block at same layer
                    layer_idx = None
                    for layer in ltp_layers:
                        if f'transformer.h.{layer}.' in key:
                            layer_idx = layer
                            break

                    if layer_idx is not None:
                        # Map keys from standard Block to BlockLTP
                        # Both have: ln_1, attn (with different signature), ln_2, mlp
                        rl_key = key  # Try direct mapping first
                        if rl_key in rl_state_dict:
                            if rl_state_dict[rl_key].shape == hybrid_state_dict[key].shape:
                                hybrid_state_dict[key] = rl_state_dict[rl_key].clone(
                                )
                                transferred_keys.append(key)
                                conversion_notes.append(
                                    f"Transferred from RL Block to LTP Block: {key}")
                            else:
                                conversion_notes.append(
                                    f"Shape mismatch for LTP layer {key}: {rl_state_dict[rl_key].shape} vs {hybrid_state_dict[key].shape}"
                                )
                        else:
                            conversion_notes.append(
                                f"LTP layer parameter (using random init): {key}")
                else:
                    missing_keys.append(key)
            else:
                missing_keys.append(key)

    # Load the transferred state dict
    missing, unexpected = hybrid_model.load_state_dict(
        hybrid_state_dict, strict=False)

    if verbose:
        print(f"\n=== Weight Transfer Summary ===")
        print(f"Transferred keys: {len(transferred_keys)}")
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Conversion notes: {len(conversion_notes)}")

        if len(missing_keys) > 0 and len(missing_keys) <= 20:
            print(f"\nMissing keys (expected for new LTP parameters):")
            for k in missing_keys[:20]:
                print(f"  - {k}")

        if len(conversion_notes) > 0:
            print(f"\nConversion notes (first 10):")
            for note in conversion_notes[:10]:
                print(f"  - {note}")

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print(f"\nCreated output directory: {output_dir}")

    # Save converted checkpoint
    converted_checkpoint = {
        'model': hybrid_model.state_dict(),
        'model_args': hybrid_args,
        'config': hybrid_args,  # For compatibility
        'iter_num': 0,  # Reset training iteration
        'best_val_loss': 1e9,  # Reset best loss
        'rl_checkpoint_source': rl_checkpoint_path,  # Track source
        'conversion_info': {
            'transferred_keys': len(transferred_keys),
            'missing_keys': len(missing_keys),
            'ltp_layers': ltp_layers,
        },
        'training_history': {
            'train_losses': [],
            'val_losses': [],
            'val_perplexities': [],
        }  # Initialize empty training history for compatibility
    }

    torch.save(converted_checkpoint, output_path)

    if verbose:
        print(f"\n=== Conversion Complete ===")
        print(f"Saved hybrid checkpoint to: {output_path}")
        print(f"Ready for LTP training!")

    return hybrid_model, converted_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description='Convert RL model checkpoint to hybrid RL+LTP architecture'
    )
    parser.add_argument(
        '--rl_checkpoint',
        type=str,
        required=True,
        help='Path to trained RL model checkpoint (e.g., out-dynamic-tr-wt2/ckpt.pt)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save converted hybrid checkpoint (e.g., out-hybrid-rl-ltp/converted_ckpt.pt)'
    )
    parser.add_argument(
        '--ltp_layers',
        type=int,
        nargs='+',
        default=[9, 10, 11],
        help='Layer indices to use LTP (default: 9 10 11)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Convert checkpoint
    convert_rl_to_hybrid(
        rl_checkpoint_path=args.rl_checkpoint,
        output_path=args.output,
        ltp_layers=tuple(args.ltp_layers),
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
