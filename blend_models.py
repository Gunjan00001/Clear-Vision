"""
Model Weight Blending Utility
==============================
Blends two PyTorch state dicts via weighted averaging to create a single
hybrid checkpoint. Useful for creating ensemble models without inference
overhead.

Usage:
    python blend_models.py teacher_best.pt cityscapes_final.pt blended_model.pt
    python blend_models.py teacher_best.pt cityscapes_final.pt blended_model.pt --alpha 0.7
"""

import argparse
import os
import sys

import torch


def blend_weights(model1_path, model2_path, output_path, alpha=0.5):
    """Blend two state dicts: output = alpha * model1 + (1-alpha) * model2."""
    if not os.path.exists(model1_path):
        print(f"Error: Could not find {model1_path}")
        return False
    if not os.path.exists(model2_path):
        print(f"Error: Could not find {model2_path}")
        return False

    print(f"Loading {model1_path}...")
    state_dict1 = torch.load(model1_path, map_location='cpu', weights_only=True)

    print(f"Loading {model2_path}...")
    state_dict2 = torch.load(model2_path, map_location='cpu', weights_only=True)

    blended = {}
    print(f"Blending weights: {alpha*100:.0f}% Model 1 / {(1-alpha)*100:.0f}% Model 2...")
    for key in state_dict1:
        if key in state_dict2:
            blended[key] = state_dict1[key] * alpha + state_dict2[key] * (1.0 - alpha)
        else:
            blended[key] = state_dict1[key]

    print(f"Saving blended model to {output_path}...")
    torch.save(blended, output_path)
    print("Done!")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Blend two PyTorch model state dicts via weighted averaging'
    )
    parser.add_argument('model1', type=str, help='Path to first model weights')
    parser.add_argument('model2', type=str, help='Path to second model weights')
    parser.add_argument('output', type=str, help='Path to save blended weights')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight for model1 (1-alpha for model2, default: 0.5)')

    args = parser.parse_args()
    if not blend_weights(args.model1, args.model2, args.output, args.alpha):
        sys.exit(1)
