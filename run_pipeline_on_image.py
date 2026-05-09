"""
Full OmniSight Pipeline: Restoration + Detection on a Single Image
===================================================================
Runs image restoration (Restormer teacher or U-Net student) followed by
YOLOv8 object detection. Saves a side-by-side comparison:
  [Original] | [Restored] | [YOLO Detected]

Usage:
    python run_pipeline_on_image.py path/to/image.jpg
    python run_pipeline_on_image.py path/to/image.jpg --model_type student --restore_weights dual_distilled_unet_best.pt
"""

import argparse
import os

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from models.restormer import Restormer
from models.autoencoder import UNet


def main():
    parser = argparse.ArgumentParser(
        description='Run OmniSight restoration + YOLO detection on a single image'
    )
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('--model_type', type=str, choices=['teacher', 'student'],
                        default='teacher',
                        help='Restoration model type (teacher=Restormer, student=UNet)')
    parser.add_argument('--restore_weights', type=str, default='teacher_best.pt',
                        help='Path to restoration model weights')
    parser.add_argument('--yolo_weights', type=str, default='yolov8n.pt',
                        help='Path to YOLO weights')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: outputs/<image>_pipeline_out.<ext>)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not os.path.exists(args.image):
        print(f"Error: Input image '{args.image}' not found.")
        return

    # 1. Load Image
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f"Error: Could not read image '{args.image}'.")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    # 2. Restoration
    print(f"Initializing '{args.model_type}' restoration model...")
    if args.model_type == 'teacher':
        restore_model = Restormer(in_channels=3, out_channels=3).to(device)
    else:
        restore_model = UNet(in_channels=6, out_channels=3).to(device)

    if os.path.exists(args.restore_weights):
        try:
            restore_model.load_state_dict(
                torch.load(args.restore_weights, map_location=device, weights_only=True)
            )
            print(f"Loaded weights from {args.restore_weights}")
        except Exception as e:
            print(f"Warning: Could not load weights: {e}")
    else:
        print(f"Warning: Weights '{args.restore_weights}' not found. Using untrained model.")

    restore_model.eval()

    print("Running restoration...")
    with torch.no_grad():
        if args.model_type == 'teacher':
            # Pad to multiple of 4
            _, _, h, w = img_tensor.shape
            pad_h = (4 - h % 4) % 4
            pad_w = (4 - w % 4) % 4
            padded = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
            restored_tensor = restore_model(padded)
            if pad_h > 0 or pad_w > 0:
                restored_tensor = restored_tensor[:, :, :h, :w]
        else:
            # Student U-Net: duplicate frame for 6-channel input
            input_6ch = torch.cat([img_tensor, img_tensor], dim=1)
            _, _, h, w = input_6ch.shape
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
            padded = torch.nn.functional.pad(input_6ch, (0, pad_w, 0, pad_h), mode='reflect')
            restored_tensor = restore_model(padded)
            if pad_h > 0 or pad_w > 0:
                restored_tensor = restored_tensor[:, :, :h, :w]

    restored_np = restored_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    restored_np = np.clip(restored_np * 255.0, 0, 255).astype(np.uint8)
    restored_bgr = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)

    # 3. Object Detection
    print(f"Running YOLO detection with '{args.yolo_weights}'...")
    try:
        yolo = YOLO(args.yolo_weights)
        results = yolo.predict(restored_bgr)
        annotated_bgr = results[0].plot()
    except Exception as e:
        print(f"Warning: YOLO failed: {e}")
        annotated_bgr = restored_bgr.copy()

    # 4. Save Output
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    if args.output:
        output_path = args.output
        if not os.path.isabs(output_path) and not os.path.dirname(output_path):
            output_path = os.path.join(output_dir, output_path)
    else:
        base, ext = os.path.splitext(os.path.basename(args.image))
        output_path = os.path.join(output_dir, f"{base}_pipeline_out{ext}")

    # Side-by-side comparison
    combined = np.hstack((img_bgr, restored_bgr, annotated_bgr))

    max_width = 1920
    if combined.shape[1] > max_width:
        scale = max_width / combined.shape[1]
        new_dim = (int(combined.shape[1] * scale), int(combined.shape[0] * scale))
        combined = cv2.resize(combined, new_dim)

    cv2.imwrite(output_path, combined)
    print(f"Saved: {output_path}")
    print(f"Layout: [Original] | [Restored] | [YOLO Detected]")


if __name__ == "__main__":
    main()
