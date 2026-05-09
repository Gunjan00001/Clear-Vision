"""
Image Corruption Simulator
===========================
Applies synthetic weather effects (rain, fog) and physical lens distortions
(out-of-focus blur, headlight bloom, contrast reduction) to clean images
for training robust restoration models.

All operations expect and produce **RGB** uint8 numpy arrays (H, W, 3).
"""

import cv2
import numpy as np
import albumentations as A


class ImageCorruptor:
    """On-the-fly image corruption pipeline for data augmentation.

    Applies a random combination of synthetic rain, fog, lens distortion,
    and photometric jitter to simulate adverse driving conditions.
    """

    def __init__(self):
        self.albu_transforms = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        ])

    def add_synthetic_rain(self, image):
        """Simulate rain streaks via thresholded vertical blur noise."""
        h, w = image.shape[:2]
        rain_drops = np.random.uniform(0, 255, (h, w))

        # Blur vertically to create streak appearance
        rain_layer = cv2.blur(rain_drops, (3, 15))
        rain_layer = cv2.threshold(rain_layer, 200, 255, cv2.THRESH_BINARY)[1]

        rain_layer = np.expand_dims(rain_layer, axis=-1)
        rain_layer = np.repeat(rain_layer, 3, axis=-1).astype(np.uint8)

        return cv2.addWeighted(image, 0.8, rain_layer, 0.2, 0)

    def add_fog(self, image):
        """Simulate fog via uniform transmission map blending."""
        fog_layer = np.full_like(image, 200, dtype=np.uint8)
        return cv2.addWeighted(image, 0.5, fog_layer, 0.5, 0)

    def apply_lens_distortion(self, image):
        """Simulate lens artifacts: out-of-focus blur, headlight bloom, contrast loss.

        Works correctly on both RGB and BGR inputs (grayscale conversion is
        only used for bright-spot detection, not for color output).
        """
        # 1. Out-of-focus water / dirt simulation
        blurred = cv2.GaussianBlur(image, (7, 7), 0)

        # 2. Headlight bloom — detect bright spots in grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        bright_mask = cv2.dilate(bright_mask, kernel, iterations=2)
        bloom = cv2.GaussianBlur(bright_mask, (41, 41), 0)

        bloom_color = cv2.cvtColor(bloom, cv2.COLOR_GRAY2BGR)
        image_with_bloom = cv2.addWeighted(blurred, 0.8, bloom_color, 0.4, 0)

        # 3. Contrast reduction (common in fog/rain)
        return cv2.convertScaleAbs(image_with_bloom, alpha=0.7, beta=30)

    def __call__(self, image):
        """Apply random corruption pipeline to *image* (RGB uint8 ndarray)."""
        img = image.copy()

        if np.random.rand() > 0.5:
            img = self.add_synthetic_rain(img)
        if np.random.rand() > 0.5:
            img = self.add_fog(img)

        # Always apply some lens distortion to close the domain gap
        img = self.apply_lens_distortion(img)

        # Photometric augmentations
        img = self.albu_transforms(image=img)['image']
        return img
