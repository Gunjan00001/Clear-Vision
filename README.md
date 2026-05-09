<p align="center">
  <h1 align="center">OmniSight Vision Pipeline</h1>
  <p align="center">
    <strong>Real-Time Adverse Weather Image Restoration for Autonomous Perception</strong>
  </p>
  <p align="center">
    <a href="#architecture">Architecture</a> •
    <a href="#key-innovations">Key Innovations</a> •
    <a href="#results">Results</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#training">Training</a> •
    <a href="#inference">Inference</a>
  </p>
</p>

---

## Problem Statement

Object detection models (YOLOv8, DETR, etc.) suffer **catastrophic accuracy drops** (up to 40% mAP degradation) when deployed in real-world adverse weather — rain, fog, low-light, and lens contamination. Existing restoration networks like Restormer achieve high-fidelity recovery but run at **~2 FPS** on edge hardware, making them impractical for real-time autonomous systems.

**OmniSight solves this** by distilling the restoration capability of heavy transformer models into a lightweight U-Net that runs at **real-time speeds** while preserving downstream detection accuracy.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OFFLINE: Knowledge Distillation                  │
│                                                                     │
│   Noisy Image ──► [Frozen Teacher 1: BDD100k Restormer]  ──┐       │
│                                                              ├─► Avg │
│   Noisy Image ──► [Frozen Teacher 2: Cityscapes Restormer] ─┘       │
│                                                    │                │
│                                          Pseudo Ground Truth        │
│                                                    │                │
│   Noisy 6-ch (T-1, T) ──► [Student U-Net] ──► Perceptual Loss      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    ONLINE: Edge Inference Pipeline                   │
│                                                                     │
│   Camera Feed ──► Temporal Buffer ──► Student U-Net (FP16)          │
│                                            │                        │
│                                     Restored Frame                  │
│                                            │                        │
│                                   YOLOv8 + ByteTrack               │
│                                            │                        │
│                                  Tracked Detections                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Innovations

### 1. Dual-Teacher Ensemble Distillation
Instead of distilling from a single model, OmniSight averages predictions from **two specialized Restormer teachers** — one trained on BDD100k (general driving) and one fine-tuned on Cityscapes (urban-specific). This ensemble pseudo-ground-truth is richer and more robust than any single teacher.

**Implementation:** [`train_dual_teacher.py`](train_dual_teacher.py) · [`train_unet_from_pretrained.py`](train_unet_from_pretrained.py)

### 2. Perceptual Distillation Loss
Standard MSE produces blurry outputs. OmniSight uses a **composite loss function** with three complementary terms:

| Component | Weight | Purpose |
|-----------|--------|---------|
| **Charbonnier Loss** | 0.6 | Pixel-level fidelity with edge preservation (smooth L1 variant) |
| **MSE Loss** | 0.3 | Stable gradient signal for convergence |
| **SSIM-Proxy Loss** | 0.1 | Structural similarity via channel-wise mean/variance matching |

**Implementation:** [`train_unet_from_pretrained.py` → `PerceptualDistillationLoss`](train_unet_from_pretrained.py)

### 3. Temporal Consistency via 6-Channel Input
Processing video frames independently causes **output flickering** that degrades downstream tracking. OmniSight concatenates frame T-1 and frame T into a **6-channel tensor**, giving the student model temporal context to enforce frame-to-frame coherence.

**Impact:** Significantly reduces ByteTrack ID switches and improves MOTA.

### 4. Weight-Space Ensemble (Model Blending)
An alternative to runtime ensemble: OmniSight provides a utility to **arithmetically average the weight tensors** of two models, producing a single hybrid checkpoint with no inference overhead.

```bash
python blend_models.py teacher_best.pt cityscapes_final.pt blended_model.pt --alpha 0.5
```

**Implementation:** [`blend_models.py`](blend_models.py)

---

## Technical Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Teacher Model** | Restormer (MDTA + GDFN) | State-of-the-art channel-wise self-attention for high-res restoration |
| **Student Model** | U-Net (6-ch input) | Lightweight CNN with skip connections — fast on edge GPUs |
| **Detection** | YOLOv8-nano + ByteTrack | Real-time object detection with multi-object tracking |
| **Training** | AdamW + Cosine Annealing + AMP | Mixed-precision training with gradient clipping and early stopping |
| **Corruption Sim** | OpenCV + Albumentations | Synthetic rain, fog, lens blur, headlight bloom |

---

## Project Structure

```
omnisight/
├── models/
│   ├── restormer.py              # Restormer: MDTA + GDFN transformer blocks
│   └── autoencoder.py            # Temporal U-Net student (6-ch → 3-ch)
├── data/
│   ├── dataset.py                # OmniSightDataset: temporal frame pairs + on-the-fly corruption
│   └── corruption.py             # Rain, fog, lens distortion, headlight bloom simulation
├── inference/
│   └── pipeline.py               # Real-time: U-Net restoration → YOLOv8 + ByteTrack
│
├── train_restore.py              # Stage 1: Train base Restormer teacher
├── train_dual_teacher.py         # Stage 2: Dual-teacher distillation (simple)
├── train_unet_from_pretrained.py # Stage 2: Advanced distillation (AMP, scheduler, checkpointing)
├── blend_models.py               # Utility: weight-space model averaging
├── compare_models.py             # Utility: visual side-by-side model comparison
├── run_image_restoration.py      # Inference: single-image Restormer restoration
├── run_pipeline_on_image.py      # Inference: full pipeline (restore → detect)
├── eval_restore_and_detect.py    # Evaluation: baseline vs OmniSight on video
├── test_shapes.py                # Smoke test: model I/O shape verification
│
├── teacher_best.pt               # Checkpoint: base Restormer (BDD100k)
├── cityscapes_final.pt           # Checkpoint: Cityscapes-finetuned Restormer
├── blended_model.pt              # Checkpoint: 50/50 weight blend
├── dual_distilled_unet_best.pt   # Checkpoint: distilled student U-Net
└── yolov8n.pt                    # Checkpoint: YOLOv8-nano
```

---

## Model Specifications

| Model | Parameters | Input | Output | Role |
|-------|-----------|-------|--------|------|
| **Restormer** (dim=32) | ~900K | `(B, 3, H, W)` | `(B, 3, H, W)` | Teacher — high-fidelity restoration |
| **U-Net** (64→1024) | ~31M | `(B, 6, H, W)` | `(B, 3, H, W)` | Student — fast temporal restoration |
| **YOLOv8-nano** | ~3.2M | `(B, 3, 640, 640)` | Detections | Downstream object detector |

---

## Quick Start

### Setup
```bash
git clone https://github.com/Gunjan00001/Clear-Vision.git
cd Clear-Vision
python -m venv venv && venv\Scripts\activate  # Windows
pip install -r requirements.txt
git lfs pull  # Download model weights
```

### Verify Installation
```bash
python test_shapes.py
```

### Run Inference
```bash
# Single-image restoration (Restormer teacher)
python run_image_restoration.py path/to/image.jpg --weights teacher_best.pt

# Full pipeline: restore + detect
python run_pipeline_on_image.py path/to/image.jpg --model_type student \
    --restore_weights dual_distilled_unet_best.pt

# Compare all checkpoints side-by-side
python compare_models.py path/to/image.jpg
```

---

## Training

### Stage 1: Train Base Teacher
```bash
python train_restore.py --data_dir ./data/train --epochs 50 --lr 2e-4
```

### Stage 2a: Dual-Teacher Distillation
```bash
python train_dual_teacher.py \
    --data_dir ./data/train \
    --teacher1_weights teacher_best.pt \
    --teacher2_weights cityscapes_final.pt \
    --epochs 50
```

### Stage 2b: Advanced Distillation (Recommended)
```bash
python train_unet_from_pretrained.py \
    --teacher_weights blended_model.pt \
    --data_dir ./data/train \
    --epochs 50 --amp --patience 15 --grad_clip 1.0
```

### Weight Blending
```bash
python blend_models.py teacher_best.pt cityscapes_final.pt blended_model.pt --alpha 0.5
```

---

## Evaluation

```bash
# Baseline vs OmniSight on video
python eval_restore_and_detect.py --video path/to/video.mp4
```

---

## Design Decisions & Trade-offs

| Decision | Alternative Considered | Rationale |
|----------|----------------------|-----------|
| U-Net over MobileNet | MobileNetV3 decoder | U-Net skip connections preserve spatial detail critical for restoration |
| Charbonnier over L1 | Standard L1/L2 | Differentiable at zero, better edge preservation in practice |
| 6-channel temporal input | Optical flow warping | Lower computational cost, no flow estimation overhead |
| Weight blending | Runtime ensemble | Zero additional inference cost; single forward pass |
| Cosine annealing | StepLR / ReduceLROnPlateau | Smoother convergence; works well with knowledge distillation |
| Channel-wise attention (MDTA) | Spatial self-attention | O(C²) vs O(N²) — critical for high-resolution inputs |

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU training)
- ~500MB disk for model checkpoints (Git LFS)

See [`requirements.txt`](requirements.txt) for full dependency list.

---

## License

This project is for academic and portfolio purposes.

---

<p align="center">
  <sub>Built with PyTorch • YOLOv8 • Restormer Architecture</sub>
</p>
