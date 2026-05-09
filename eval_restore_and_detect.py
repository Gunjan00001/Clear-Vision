"""
End-to-End Evaluation: Baseline vs OmniSight Pipeline
======================================================
Compares detection counts between:
  1. Baseline:  Corrupted video → YOLOv8 + ByteTrack
  2. OmniSight: Corrupted video → U-Net restoration → YOLOv8 + ByteTrack

Usage:
    python eval_restore_and_detect.py --video path/to/video.mp4
"""

import argparse
import os

import cv2
import numpy as np
from ultralytics import YOLO

from inference.pipeline import OmniSightPipeline


def evaluate_baseline(video_path, yolo_weights="yolov8n.pt"):
    """Run YOLO directly on corrupted frames (no restoration)."""
    print("Evaluating Baseline Pipeline (Corrupted → YOLOv8 + ByteTrack)")
    yolo = YOLO(yolo_weights)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_detections = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo.track(frame, tracker="bytetrack.yaml", persist=True)
        if results[0].boxes is not None:
            total_detections += len(results[0].boxes)
        frame_count += 1

    cap.release()
    print(f"Baseline: {frame_count} frames, {total_detections} total detections.")
    return total_detections


def evaluate_omnisight(video_path):
    """Run YOLO on U-Net-restored frames."""
    print("Evaluating OmniSight Pipeline (Corrupted → U-Net → YOLOv8 + ByteTrack)")
    pipeline = OmniSightPipeline()

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_detections = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        _, results = pipeline.process_frame(frame)
        if results[0].boxes is not None:
            total_detections += len(results[0].boxes)
        frame_count += 1

    cap.release()
    print(f"OmniSight: {frame_count} frames, {total_detections} total detections.")
    return total_detections


def main(args):
    if not os.path.exists(args.video):
        print(f"Video '{args.video}' not found. Creating dummy video for smoke test...")
        h, w = 256, 256
        out = cv2.VideoWriter("dummy.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
        for _ in range(30):
            out.write(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
        out.release()
        args.video = "dummy.mp4"

    baseline = evaluate_baseline(args.video)
    omnisight = evaluate_omnisight(args.video)

    print("\n--- Evaluation Results ---")
    print(f"Baseline Detections:  {baseline}")
    print(f"OmniSight Detections: {omnisight}")
    print("(For mAP/MOTA metrics, ground-truth annotations are required.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compare baseline vs OmniSight pipeline on video'
    )
    parser.add_argument('--video', type=str, default='test_video.mp4',
                        help='Path to corrupted test video')
    main(parser.parse_args())
