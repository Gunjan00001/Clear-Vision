"""
Single-Image Restormer Inference
=================================
Runs a trained Restormer model on a single image and saves the restored output.

Usage:
    python run_image_restoration.py path/to/image.jpg
    python run_image_restoration.py path/to/image.jpg --weights cityscapes_final.pt
    python run_image_restoration.py path/to/image.jpg --output restored.jpg
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

from models.restormer import Restormer


def load_image(path):
    """Load an image as a normalized (1, C, H, W) RGB float tensor."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


def save_image(tensor, out_path):
    """Save a (1, C, H, W) RGB tensor as a BGR image file."""
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, img)


def main():
    parser = argparse.ArgumentParser(
        description='Run Restormer image restoration on a single image'
    )
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('--weights', type=str, default='teacher_best.pt',
                        help='Path to model weights (default: teacher_best.pt)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: <image>_restored.<ext>)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Restormer(in_channels=3, out_channels=3).to(device)

    weight_path = args.weights
    if not os.path.isabs(weight_path):
        weight_path = os.path.join(os.path.dirname(__file__) or '.', weight_path)
    if not os.path.exists(weight_path):
        print(f"Error: Weight file not found: {weight_path}")
        sys.exit(1)

    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        img_tensor = load_image(args.image).to(device)

        # Pad to multiple of 4 (model downsamples 2× twice)
        _, _, h, w = img_tensor.shape
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        restored = model(img_tensor)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            restored = restored[:, :, :h, :w]

    if args.output:
        out_path = args.output
    else:
        base, ext = os.path.splitext(args.image)
        out_path = f"{base}_restored{ext}"

    save_image(restored, out_path)
    print(f"Restored image saved to: {out_path}")


if __name__ == "__main__":
    main()
