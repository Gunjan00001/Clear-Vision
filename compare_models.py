"""
Model Comparison Tool
=====================
Runs an input image through multiple Restormer checkpoints side-by-side
and saves a labeled comparison strip.

Usage:
    python compare_models.py path/to/image.jpg
    python compare_models.py path/to/image.jpg --output comparison.jpg
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

from models.restormer import Restormer
from run_image_restoration import load_image


def save_comparison(images, labels, out_path):
    """Create a horizontal strip with labeled images and save to disk."""
    labeled = []
    for img, label in zip(images, labels):
        img_copy = img.copy()
        cv2.rectangle(img_copy, (5, 5), (250, 45), (0, 0, 0), -1)
        cv2.putText(img_copy, label, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        labeled.append(img_copy)

    combined = cv2.hconcat(labeled)
    cv2.imwrite(out_path, combined)


def main():
    parser = argparse.ArgumentParser(
        description='Compare restoration output across multiple model checkpoints'
    )
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: <image>_comparison.<ext>)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    models_to_test = {
        '1. Base Teacher': 'teacher_best.pt',
        '2. Cityscapes': 'cityscapes_final.pt',
        '3. Blended 50/50': 'blended_model.pt',
    }

    # Load and resize original
    orig_img = cv2.imread(args.image)
    if orig_img is None:
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    h, w = orig_img.shape[:2]
    new_w = 640
    new_h = int(h * (new_w / w))
    orig_img = cv2.resize(orig_img, (new_w, new_h))

    results = [orig_img]
    labels = ['Original']

    model = Restormer(in_channels=3, out_channels=3).to(device)

    for name, weight_file in models_to_test.items():
        if not os.path.exists(weight_file):
            print(f"Skipping {name}: {weight_file} not found.")
            continue

        print(f"Processing through {name}...")
        model.load_state_dict(torch.load(weight_file, map_location=device, weights_only=True))
        model.eval()

        with torch.no_grad():
            img_tensor = load_image(args.image).to(device)
            _, _, th, tw = img_tensor.shape
            pad_h = (4 - th % 4) % 4
            pad_w = (4 - tw % 4) % 4
            img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

            restored = model(img_tensor)

            if pad_h > 0 or pad_w > 0:
                restored = restored[:, :, :th, :tw]

        out_np = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_np = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)
        out_np = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
        out_np = cv2.resize(out_np, (new_w, new_h))

        results.append(out_np)
        labels.append(name)

    if args.output:
        out_path = args.output
    else:
        base, ext = os.path.splitext(args.image)
        out_path = f"{base}_comparison{ext}"

    save_comparison(results, labels, out_path)
    print(f"\nComparison saved to: {out_path}")


if __name__ == '__main__':
    main()
