"""
OmniSight Dataset
=================
Loads clean images from BDD100k (and optionally real-rain captures), applies
synthetic corruptions on-the-fly, and returns 6-channel temporal frame pairs
for student model training.
"""

import os
import warnings

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .corruption import ImageCorruptor

# Supported image extensions
_IMG_EXTENSIONS = frozenset(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))


def _scan_images(directory):
    """Return sorted list of image filenames in *directory*."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        entry.name for entry in os.scandir(directory)
        if entry.is_file() and os.path.splitext(entry.name)[1].lower() in _IMG_EXTENSIONS
    )


class OmniSightDataset(Dataset):
    """Mixed BDD100k + Real-Rain dataset with on-the-fly corruption.

    Args:
        bdd100k_dir:   Path to BDD100k training images.
        real_rain_dir:  Path to real-rain captures (optional, can be empty string).
        target_size:    Spatial size to resize all images to (H, W).
        real_rain_prob: Probability of sampling from real-rain vs BDD100k.
        transform:      Optional additional torchvision-style transform.
    """

    def __init__(self, bdd100k_dir, real_rain_dir="", target_size=(256, 256),
                 real_rain_prob=0.2, transform=None):
        self.bdd100k_dir = bdd100k_dir
        self.real_rain_dir = real_rain_dir
        self.target_size = target_size
        self.real_rain_prob = real_rain_prob
        self.transform = transform
        self.corruptor = ImageCorruptor()

        self.bdd_files = _scan_images(bdd100k_dir)
        self.real_files = _scan_images(real_rain_dir) if real_rain_dir else []

        total = max(len(self.bdd_files), len(self.real_files))
        if total == 0:
            warnings.warn(
                f"No images found in '{bdd100k_dir}' or '{real_rain_dir}'. "
                "The dataset will return synthetic noise tensors — this is only "
                "suitable for smoke-testing, not real training.",
                UserWarning, stacklevel=2,
            )
            self.total_samples = 100  # Fallback for CI / smoke-test
        else:
            self.total_samples = total * 2

    def __len__(self):
        return self.total_samples

    def _get_frame_pair(self, file_list, dir_path, idx):
        """Load consecutive frames (T-1, T) and resize to target size."""
        actual_idx = idx % max(1, len(file_list))
        prev_idx = max(0, actual_idx - 1)

        path_t = os.path.join(dir_path, file_list[actual_idx])
        path_prev = os.path.join(dir_path, file_list[prev_idx])

        img_t = cv2.imread(path_t)
        img_prev = cv2.imread(path_prev)

        # Graceful fallback for unreadable files
        if img_t is None:
            img_t = np.zeros((*self.target_size, 3), dtype=np.uint8)
        if img_prev is None:
            img_prev = np.zeros((*self.target_size, 3), dtype=np.uint8)

        img_t = cv2.resize(img_t, self.target_size)
        img_prev = cv2.resize(img_prev, self.target_size)

        img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
        img_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2RGB)

        return img_prev, img_t

    def __getitem__(self, idx):
        use_real = np.random.rand() < self.real_rain_prob

        if use_real and len(self.real_files) > 0:
            img_prev, img_clean = self._get_frame_pair(self.real_files, self.real_rain_dir, idx)
            img_noisy = self.corruptor.apply_lens_distortion(img_clean)
            img_prev_noisy = self.corruptor.apply_lens_distortion(img_prev)
        elif len(self.bdd_files) > 0:
            img_prev, img_clean = self._get_frame_pair(self.bdd_files, self.bdd100k_dir, idx)
            img_noisy = self.corruptor(img_clean)
            img_prev_noisy = self.corruptor(img_prev)
        else:
            # Synthetic noise fallback (smoke-testing only)
            img_clean = np.random.randint(0, 255, (*self.target_size, 3), dtype=np.uint8)
            img_prev = img_clean.copy()
            img_noisy = self.corruptor(img_clean)
            img_prev_noisy = self.corruptor(img_prev)

        # Normalize to [0, 1] and convert to (C, H, W) tensors
        img_noisy = torch.from_numpy(img_noisy).float().permute(2, 0, 1) / 255.0
        img_prev_noisy = torch.from_numpy(img_prev_noisy).float().permute(2, 0, 1) / 255.0
        img_clean = torch.from_numpy(img_clean).float().permute(2, 0, 1) / 255.0

        # 6-channel input: [T-1, T] concatenated along channel dim
        input_6ch = torch.cat([img_prev_noisy, img_noisy], dim=0)

        return {
            'input_6ch': input_6ch,
            'noisy_t': img_noisy,
            'clean_t': img_clean,
        }
