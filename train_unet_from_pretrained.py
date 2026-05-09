"""
Train U-Net Student via Knowledge Distillation from Pretrained Teacher
======================================================================
Advanced training pipeline with:
  - Perceptual Distillation Loss (Charbonnier + MSE + SSIM-proxy)
  - Automatic Mixed Precision (AMP)
  - Cosine Annealing LR scheduler
  - Gradient clipping & early stopping
  - Train/validation split with checkpointing

Pretrained teacher models:
  - teacher_best.pt      : Base Restormer (BDD100k)
  - cityscapes_final.pt  : Restormer fine-tuned on Cityscapes
  - blended_model.pt     : 50/50 weight blend of the above

Usage:
    python train_unet_from_pretrained.py --teacher_weights blended_model.pt
    python train_unet_from_pretrained.py --teacher_weights teacher_best.pt --epochs 100
    python train_unet_from_pretrained.py --resume distilled_unet_checkpoint.pt
"""

import argparse
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from data.dataset import OmniSightDataset
from models.autoencoder import UNet
from models.restormer import Restormer


# ──────────────────────────────────────────────────────────────────────────────
# Loss Functions
# ──────────────────────────────────────────────────────────────────────────────

class CharbonnierLoss(nn.Module):
    """Smooth L1 alternative — better for image restoration than MSE."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class PerceptualDistillationLoss(nn.Module):
    """Combined loss for distillation:
      - Charbonnier  : pixel-level fidelity
      - MSE          : smooth gradient signal
      - SSIM-proxy   : structural similarity via mean/variance matching
    """

    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.charbonnier = CharbonnierLoss()
        self.mse = nn.MSELoss()

    def _ssim_proxy(self, pred, target):
        """Lightweight structural similarity proxy using channel statistics."""
        mu_pred = pred.mean(dim=[2, 3], keepdim=True)
        mu_target = target.mean(dim=[2, 3], keepdim=True)
        var_pred = ((pred - mu_pred) ** 2).mean(dim=[2, 3], keepdim=True)
        var_target = ((target - mu_target) ** 2).mean(dim=[2, 3], keepdim=True)
        return self.mse(mu_pred, mu_target) + self.mse(var_pred, var_target)

    def forward(self, pred, target):
        return (self.alpha * self.charbonnier(pred, target)
                + self.beta * self.mse(pred, target)
                + self.gamma * self._ssim_proxy(pred, target))


# ──────────────────────────────────────────────────────────────────────────────
# Training Utilities
# ──────────────────────────────────────────────────────────────────────────────

def count_parameters(model):
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_time(seconds):
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def print_banner(teacher_path, device, student_params, teacher_params, dataset_size):
    """Print a training summary banner."""
    print("\n" + "=" * 70)
    print("  🎓  U-Net Student Training via Knowledge Distillation")
    print("=" * 70)
    print(f"  Teacher weights : {teacher_path}")
    print(f"  Device          : {device} ", end="")
    if device.type == 'cuda':
        print(f"({torch.cuda.get_device_name(0)})")
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  VRAM            : {vram:.1f} GB")
    else:
        print("(⚠ CPU – training will be slow)")
    print(f"  Teacher params  : {teacher_params:,}")
    print(f"  Student params  : {student_params:,}  "
          f"({student_params / teacher_params * 100:.1f}% of teacher)")
    print(f"  Dataset size    : {dataset_size:,} samples")
    print("=" * 70 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = args.amp and device.type == 'cuda'

    if not os.path.isdir(args.data_dir):
        print(f"WARNING: Data directory '{args.data_dir}' not found. "
              "Dataset will fall back to synthetic noise.")

    # ── Dataset ───────────────────────────────────────────────────────────
    full_dataset = OmniSightDataset(
        bdd100k_dir=args.data_dir,
        real_rain_dir=args.real_rain_dir,
    )

    # Train / Validation split (90/10)
    total = len(full_dataset)
    val_size = max(1, int(total * 0.1))
    train_size = total - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    # ── Teacher (frozen) ──────────────────────────────────────────────────
    teacher = Restormer(in_channels=3, out_channels=3, dim=32).to(device)
    if os.path.exists(args.teacher_weights):
        teacher.load_state_dict(
            torch.load(args.teacher_weights, map_location=device, weights_only=True)
        )
        print(f"✅ Loaded teacher weights: {args.teacher_weights}")
    else:
        print(f"❌ WARNING: {args.teacher_weights} not found – using random teacher!")

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # ── Student (trainable) ───────────────────────────────────────────────
    student = UNet(in_channels=6, out_channels=3).to(device)

    # Optionally resume from a previous checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            student.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"✅ Resumed from checkpoint: {args.resume} (epoch {start_epoch})")
        else:
            student.load_state_dict(checkpoint)
            print(f"✅ Loaded student weights: {args.resume}")

    # ── Optimizer, Scheduler, Loss ────────────────────────────────────────
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )
    criterion = PerceptualDistillationLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ── Print info ────────────────────────────────────────────────────────
    teacher_total, _ = count_parameters(teacher)
    student_total, student_trainable = count_parameters(student)
    print_banner(
        args.teacher_weights, device,
        student_trainable, teacher_total, total,
    )
    print(f"  Hyperparameters:")
    print(f"    Batch size    : {args.batch_size}")
    print(f"    Epochs        : {args.epochs}")
    print(f"    Learning rate : {args.lr}")
    print(f"    Loss          : Perceptual Distillation (Charb + MSE + SSIM-proxy)")
    print(f"    Scheduler     : Cosine Annealing")
    print(f"    AMP           : {'Enabled' if use_amp else 'Disabled'}")
    print(f"    Grad clip     : {args.grad_clip}")
    print()

    # ── Training ──────────────────────────────────────────────────────────
    patience_counter = 0
    training_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        student.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            input_6ch = batch['input_6ch'].to(device, non_blocking=True)
            noisy_t = batch['noisy_t'].to(device, non_blocking=True)

            # Teacher forward (frozen)
            with torch.no_grad():
                teacher_output = teacher(noisy_t)

            # Student forward + loss
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    student_output = student(input_6ch)
                    loss = criterion(student_output, teacher_output)
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                student_output = student(input_6ch)
                loss = criterion(student_output, teacher_output)
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                optimizer.step()

            train_loss += loss.item()

            if batch_idx % args.log_interval == 0:
                elapsed = format_time(time.time() - training_start)
                lr_now = optimizer.param_groups[0]['lr']
                print(
                    f"  [{elapsed}] Epoch {epoch+1}/{args.epochs} "
                    f"| Step {batch_idx:>4d}/{len(train_loader)} "
                    f"| Loss: {loss.item():.6f} "
                    f"| LR: {lr_now:.2e}"
                )

        avg_train_loss = train_loss / max(1, len(train_loader))

        # ── Validation ────────────────────────────────────────────────────
        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_6ch = batch['input_6ch'].to(device, non_blocking=True)
                noisy_t = batch['noisy_t'].to(device, non_blocking=True)

                teacher_output = teacher(noisy_t)
                student_output = student(input_6ch)
                loss = criterion(student_output, teacher_output)
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        epoch_time = time.time() - epoch_start

        scheduler.step()

        print(f"\n{'─' * 60}")
        print(f"  Epoch {epoch+1}/{args.epochs} complete in {format_time(epoch_time)}")
        print(f"    Train Loss : {avg_train_loss:.6f}")
        print(f"    Val   Loss : {avg_val_loss:.6f}")
        print(f"    Best  Loss : {best_val_loss:.6f}")

        # ── Checkpointing ─────────────────────────────────────────────────
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint_data, 'distilled_unet_checkpoint.pt')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(student.state_dict(), 'distilled_unet_best.pt')
            print(f"    ✅ New best model saved! (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"    ⏳ No improvement ({patience_counter}/{args.patience})")

        print(f"{'─' * 60}\n")

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"  🛑 Early stopping after {args.patience} epochs without improvement.")
            break

    # ── Final Summary ─────────────────────────────────────────────────────
    total_time = time.time() - training_start
    print("\n" + "=" * 70)
    print("  🏁  Training Complete!")
    print("=" * 70)
    print(f"  Total time      : {format_time(total_time)}")
    print(f"  Best val loss   : {best_val_loss:.6f}")
    print(f"  Best model saved: distilled_unet_best.pt")
    print(f"  Last checkpoint : distilled_unet_checkpoint.pt")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train U-Net student via distillation from pretrained teacher'
    )

    # Data
    parser.add_argument('--data_dir', type=str, default='./data/train',
                        help='Path to training images directory')
    parser.add_argument('--real_rain_dir', type=str, default='',
                        help='Path to real-rain dataset (optional)')

    # Model
    parser.add_argument('--teacher_weights', type=str, default='blended_model.pt',
                        help='Path to pretrained teacher weights')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to student checkpoint to resume from')

    # Training
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Max gradient norm (0 to disable)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (0 to disable)')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use automatic mixed precision on GPU')
    parser.add_argument('--no-amp', action='store_false', dest='amp',
                        help='Disable AMP')

    # Misc
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N batches')

    main(parser.parse_args())
