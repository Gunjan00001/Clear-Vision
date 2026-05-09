"""
OmniSight Inference Pipeline
==============================
Combines U-Net restoration and YOLOv8 + ByteTrack for real-time video
processing. The pipeline:
  1. Preprocesses frames to RGB tensors
  2. Creates 6-channel temporal input (T-1, T)
  3. Runs U-Net restoration (FP16 on GPU)
  4. Passes restored frame to YOLOv8 + ByteTrack
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from models.autoencoder import UNet


class OmniSightPipeline:
    """End-to-end video restoration + tracking pipeline.

    Args:
        unet_weights: Path to trained U-Net student weights.
        yolo_weights: Path to YOLOv8 weights.
    """

    def __init__(self, unet_weights="distilled_unet_best.pt", yolo_weights="yolov8n.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load Student U-Net
        self.unet = UNet(in_channels=6, out_channels=3).to(self.device)
        try:
            self.unet.load_state_dict(
                torch.load(unet_weights, map_location=self.device, weights_only=True)
            )
            print(f"Loaded U-Net weights: {unet_weights}")
        except Exception as e:
            print(f"Warning: Could not load U-Net weights: {e}")
        self.unet.eval()

        # Load YOLOv8
        self.yolo = YOLO(yolo_weights)
        self.prev_frame_tensor = None

    def preprocess(self, frame):
        """Convert BGR numpy frame to normalized (1, 3, H, W) RGB tensor."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(self.device)

    def postprocess(self, tensor):
        """Convert (1, 3, H, W) RGB tensor back to BGR numpy."""
        img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def process_frame(self, current_frame):
        """Run restoration + detection on a single frame.

        Returns:
            restored_frame: BGR numpy array of the restored image.
            results: YOLOv8 tracking results.
        """
        curr_tensor = self.preprocess(current_frame)

        if self.prev_frame_tensor is None:
            self.prev_frame_tensor = curr_tensor.clone()

        # 6-channel temporal input
        input_6ch = torch.cat([self.prev_frame_tensor, curr_tensor], dim=1)

        # U-Net restoration (FP16 on GPU for speed)
        with torch.no_grad():
            with torch.amp.autocast(self.device.type, dtype=torch.float16):
                restored_tensor = self.unet(input_6ch)

        self.prev_frame_tensor = curr_tensor.clone()
        restored_frame = self.postprocess(restored_tensor)

        # YOLOv8 + ByteTrack
        results = self.yolo.track(restored_frame, tracker="bytetrack.yaml", persist=True)
        return restored_frame, results

    def run_video(self, video_path, output_path="output.mp4"):
        """Process an entire video through the pipeline.

        Args:
            video_path:  Path to input video file.
            output_path: Path for the output comparison video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video: {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            restored_frame, results = self.process_frame(frame)
            annotated_frame = results[0].plot()

            # Side-by-side: original | restored+tracked
            h, w = frame.shape[:2]
            half_w = w // 2
            orig_resized = cv2.resize(frame, (half_w, h))
            anno_resized = cv2.resize(annotated_frame, (half_w, h))
            combined = np.hstack((orig_resized, anno_resized))

            out.write(combined)
            cv2.imshow("OmniSight (Left: Original | Right: Restored+Tracked)", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Video saved to: {output_path}")
