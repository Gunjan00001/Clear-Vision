"""
Train Restormer Teacher Model
==============================
Trains the base Restormer teacher on clean/corrupted image pairs using
Charbonnier loss. Saves the best checkpoint as ``teacher_best.pt``.

Usage:
    python train_restore.py --data_dir ./data/train --epochs 50
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import OmniSightDataset
from models.restormer import Restormer


class CharbonnierLoss(nn.Module):
    """Charbonnier loss — a smooth L1 variant that better preserves edges."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not os.path.isdir(args.data_dir):
        print(f"WARNING: Data directory '{args.data_dir}' not found. "
              "Dataset will fall back to synthetic noise (smoke-test only).")

    dataset = OmniSightDataset(bdd100k_dir=args.data_dir, real_rain_dir=args.real_rain_dir)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
    )

    model = Restormer(in_channels=3, out_channels=3, dim=32).to(device)
    criterion = CharbonnierLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            noisy = batch['noisy_t'].to(device)
            clean = batch['clean_t'].to(device)

            optimizer.zero_grad(set_to_none=True)
            restored = model(noisy)
            loss = criterion(restored, clean)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], "
                      f"Step [{batch_idx}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'teacher_best.pt')
            print(f"Saved best model with loss {best_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Restormer teacher model on clean/corrupted image pairs'
    )
    parser.add_argument('--data_dir', type=str, default='./data/train',
                        help='Path to training images directory')
    parser.add_argument('--real_rain_dir', type=str, default='',
                        help='Path to real-rain dataset (optional)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    main(args)
