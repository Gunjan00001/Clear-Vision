"""
Dual-Teacher Knowledge Distillation
=====================================
Trains a lightweight U-Net student by averaging predictions from two frozen
Restormer teachers (base + Cityscapes-finetuned). Saves the best student
checkpoint as ``dual_distilled_unet_best.pt``.

Usage:
    python train_dual_teacher.py --data_dir ./data/train --epochs 50
    python train_dual_teacher.py --teacher1_weights teacher_best.pt --teacher2_weights cityscapes_final.pt
"""

import argparse
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import OmniSightDataset
from models.autoencoder import UNet
from models.restormer import Restormer


def _timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_label = f"{device}"
    if device.type == 'cuda':
        device_label += f" ({torch.cuda.get_device_name(0)})"
    print(f"[{_timestamp()}] Device: {device_label}")

    if not os.path.isdir(args.data_dir):
        print(f"WARNING: Data directory '{args.data_dir}' not found. "
              "Dataset will fall back to synthetic noise.")

    # Dataset
    dataset = OmniSightDataset(bdd100k_dir=args.data_dir, real_rain_dir="")
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == 'cuda'),
    )
    print(f"[{_timestamp()}] Dataset loaded with {len(dataset)} samples.")

    # Load BOTH teacher models
    teacher1 = Restormer(in_channels=3, out_channels=3, dim=32).to(device)
    teacher2 = Restormer(in_channels=3, out_channels=3, dim=32).to(device)

    if not os.path.exists(args.teacher1_weights) or not os.path.exists(args.teacher2_weights):
        print(f"[{_timestamp()}] ERROR: Could not find one or both teacher weights:")
        print(f"  teacher1: {args.teacher1_weights} (exists={os.path.exists(args.teacher1_weights)})")
        print(f"  teacher2: {args.teacher2_weights} (exists={os.path.exists(args.teacher2_weights)})")
        return

    teacher1.load_state_dict(torch.load(args.teacher1_weights, map_location=device, weights_only=True))
    teacher2.load_state_dict(torch.load(args.teacher2_weights, map_location=device, weights_only=True))
    print(f"[{_timestamp()}] Loaded both teacher models.")

    # Freeze teachers
    teacher1.eval()
    teacher2.eval()
    for param in teacher1.parameters():
        param.requires_grad = False
    for param in teacher2.parameters():
        param.requires_grad = False

    # Student
    student = UNet(in_channels=6, out_channels=3).to(device)
    print(f"[{_timestamp()}] Student U-Net initialized.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=args.lr)

    best_loss = float('inf')

    print(f"\n[{_timestamp()}] Starting Dual-Teacher Distillation...\n")

    for epoch in range(args.epochs):
        student.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            input_6ch = batch['input_6ch'].to(device)
            noisy_t = batch['noisy_t'].to(device)

            # Dual teacher prediction → averaged ensemble target
            with torch.no_grad():
                t1_clean = teacher1(noisy_t)
                t2_clean = teacher2(noisy_t)
                ensemble_target = (t1_clean + t2_clean) / 2.0

            # Student training
            optimizer.zero_grad(set_to_none=True)
            student_out = student(input_6ch)
            loss = criterion(student_out, ensemble_target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"[{_timestamp()}] Epoch [{epoch+1}/{args.epochs}], "
                      f"Step [{batch_idx}/{len(dataloader)}], "
                      f"Loss: {loss.item():.5f}")

        avg_loss = epoch_loss / max(1, len(dataloader))
        epoch_time = time.time() - start_time
        print(f"[{_timestamp()}] Epoch {epoch+1} complete in {epoch_time:.1f}s — "
              f"Avg Loss: {avg_loss:.5f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(student.state_dict(), 'dual_distilled_unet_best.pt')
            print(f"[{_timestamp()}] * New Best Model Saved (Loss: {best_loss:.5f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Dual-Teacher Distillation: train U-Net student from two Restormer teachers'
    )
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_dir', type=str, default='./data/train',
                        help='Path to training images directory')
    parser.add_argument('--teacher1_weights', type=str, default='teacher_best.pt')
    parser.add_argument('--teacher2_weights', type=str, default='cityscapes_final.pt')
    main(parser.parse_args())
