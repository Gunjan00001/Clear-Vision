"""Smoke test: verify model forward pass shapes are correct."""

import torch
from models.restormer import Restormer
from models.autoencoder import UNet


def test_shapes():
    print("Testing Restormer shapes...")
    restormer = Restormer(in_channels=3, out_channels=3, dim=32)
    dummy = torch.randn(1, 3, 256, 256)
    out = restormer(dummy)
    assert out.shape == (1, 3, 256, 256), f"Restormer shape mismatch: {out.shape}"
    print(f"  Restormer OK: {list(dummy.shape)} -> {list(out.shape)}")

    print("Testing UNet shapes...")
    unet = UNet(in_channels=6, out_channels=3)
    dummy_6ch = torch.randn(1, 6, 256, 256)
    out_unet = unet(dummy_6ch)
    assert out_unet.shape == (1, 3, 256, 256), f"UNet shape mismatch: {out_unet.shape}"
    print(f"  UNet OK: {list(dummy_6ch.shape)} -> {list(out_unet.shape)}")

    print("\nAll shape tests passed!")


if __name__ == "__main__":
    test_shapes()
