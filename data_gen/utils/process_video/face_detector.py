"""
Face Detection and Tracking Module for Arbitrary Video Lipsync

This module provides face detection and tracking capabilities to enable
lipsyncing on videos where the face is not necessarily the dominant element
in the frame. It supports:
- Any video aspect ratio (not just square)
- Faces of various sizes in the frame
- Temporal consistency via tracking and smoothing
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional, Tuple
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os


@dataclass
class BoundingBox:
    """Face bounding box with coordinates and confidence"""
    x1: int  # left
    y1: int  # top
    x2: int  # right
    y2: int  # bottom
    confidence: float = 1.0

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def expand(self, margin: float) -> 'BoundingBox':
        """Expand bounding box by a margin factor"""
        w_margin = int(self.width * margin)
        h_margin = int(self.height * margin)
        return BoundingBox(
            x1=self.x1 - w_margin,
            y1=self.y1 - h_margin,
            x2=self.x2 + w_margin,
            y2=self.y2 + h_margin,
            confidence=self.confidence
        )

    def clip(self, img_width: int, img_height: int) -> 'BoundingBox':
        """Clip bounding box to image boundaries"""
        return BoundingBox(
            x1=max(0, self.x1),
            y1=max(0, self.y1),
            x2=min(img_width, self.x2),
            y2=min(img_height, self.y2),
            confidence=self.confidence
        )

    def to_square(self) -> 'BoundingBox':
        """Convert to square bounding box (expand to largest dimension)"""
        size = max(self.width, self.height)
        cx, cy = self.center
        half = size // 2
        return BoundingBox(
            x1=cx - half,
            y1=cy - half,
            x2=cx + half,
            y2=cy + half,
            confidence=self.confidence
        )


@dataclass
class CropRegion:
    """Stable crop region for a video sequence"""
    x1: int
    y1: int
    x2: int
    y2: int
    original_width: int
    original_height: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def size(self) -> Tuple[int, int]:
        return (self.width, self.height)


class FaceDetector:
    """
    Face detector and tracker using MediaPipe Face Detection.

    Provides stable face tracking across video frames with temporal smoothing.
    """

    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Initialize the face detector.

        Args:
            min_detection_confidence: Minimum confidence for face detection
        """
        self.min_detection_confidence = min_detection_confidence
        self._init_detector()

    def _init_detector(self):
        """Initialize MediaPipe face detector"""
        # Use MediaPipe Face Detection (faster than Face Mesh for just detection)
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for short-range (2m), 1 for full-range (5m)
            min_detection_confidence=self.min_detection_confidence
        )

    def detect_face_bbox(self, frame: np.ndarray) -> Optional[BoundingBox]:
        """
        Detect face bounding box in a single frame.

        Args:
            frame: RGB image as numpy array [H, W, 3]

        Returns:
            BoundingBox if face detected, None otherwise
        """
        h, w = frame.shape[:2]

        # MediaPipe requires RGB
        if frame.shape[-1] == 3:
            rgb_frame = frame
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.detector.process(rgb_frame)

        if not results.detections:
            return None

        # Get the detection with highest confidence
        best_detection = max(results.detections,
                            key=lambda d: d.score[0])

        bbox = best_detection.location_data.relative_bounding_box

        # Convert relative coordinates to absolute
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)

        return BoundingBox(
            x1=x1, y1=y1, x2=x2, y2=y2,
            confidence=best_detection.score[0]
        )

    def detect_faces_batch(self, frames: np.ndarray) -> List[Optional[BoundingBox]]:
        """
        Detect faces in a batch of frames.

        Args:
            frames: Video frames [T, H, W, 3]

        Returns:
            List of BoundingBox (or None for frames without detection)
        """
        return [self.detect_face_bbox(frame) for frame in frames]

    def track_faces(self, frames: np.ndarray,
                    smooth_window: int = 5) -> List[BoundingBox]:
        """
        Track face across video frames with temporal smoothing.

        Args:
            frames: Video frames [T, H, W, 3]
            smooth_window: Window size for temporal smoothing

        Returns:
            List of smoothed BoundingBox for each frame
        """
        # First pass: detect faces in all frames
        raw_bboxes = self.detect_faces_batch(frames)

        # Handle missing detections via interpolation
        bboxes = self._interpolate_missing(raw_bboxes, frames.shape[1], frames.shape[2])

        # Apply temporal smoothing
        smoothed = self._temporal_smooth(bboxes, smooth_window)

        return smoothed

    def _interpolate_missing(self, bboxes: List[Optional[BoundingBox]],
                             img_height: int, img_width: int) -> List[BoundingBox]:
        """
        Interpolate missing face detections.

        Uses linear interpolation between valid detections.
        """
        result = []
        n = len(bboxes)

        # Find first valid detection
        first_valid = None
        for i, bbox in enumerate(bboxes):
            if bbox is not None:
                first_valid = i
                break

        if first_valid is None:
            # No faces detected at all - use center of frame
            cx, cy = img_width // 2, img_height // 2
            size = min(img_width, img_height) // 2
            default_bbox = BoundingBox(
                x1=cx - size//2, y1=cy - size//2,
                x2=cx + size//2, y2=cy + size//2,
                confidence=0.0
            )
            return [default_bbox] * n

        # Fill in missing detections
        for i in range(n):
            if bboxes[i] is not None:
                result.append(bboxes[i])
            else:
                # Find prev and next valid
                prev_idx = None
                next_idx = None

                for j in range(i-1, -1, -1):
                    if bboxes[j] is not None:
                        prev_idx = j
                        break

                for j in range(i+1, n):
                    if bboxes[j] is not None:
                        next_idx = j
                        break

                if prev_idx is not None and next_idx is not None:
                    # Interpolate
                    t = (i - prev_idx) / (next_idx - prev_idx)
                    prev_box = bboxes[prev_idx]
                    next_box = bboxes[next_idx]

                    interpolated = BoundingBox(
                        x1=int(prev_box.x1 + t * (next_box.x1 - prev_box.x1)),
                        y1=int(prev_box.y1 + t * (next_box.y1 - prev_box.y1)),
                        x2=int(prev_box.x2 + t * (next_box.x2 - prev_box.x2)),
                        y2=int(prev_box.y2 + t * (next_box.y2 - prev_box.y2)),
                        confidence=0.5
                    )
                    result.append(interpolated)
                elif prev_idx is not None:
                    # Use previous
                    result.append(bboxes[prev_idx])
                elif next_idx is not None:
                    # Use next
                    result.append(bboxes[next_idx])
                else:
                    # Should not happen
                    raise ValueError("No valid face detection found")

        return result

    def _temporal_smooth(self, bboxes: List[BoundingBox],
                         window: int) -> List[BoundingBox]:
        """
        Apply temporal smoothing to bounding boxes.

        Uses moving average filter for smooth tracking.
        """
        n = len(bboxes)
        if n == 0:
            return bboxes

        half_window = window // 2
        smoothed = []

        for i in range(n):
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)

            # Average coordinates
            x1_avg = sum(b.x1 for b in bboxes[start:end]) / (end - start)
            y1_avg = sum(b.y1 for b in bboxes[start:end]) / (end - start)
            x2_avg = sum(b.x2 for b in bboxes[start:end]) / (end - start)
            y2_avg = sum(b.y2 for b in bboxes[start:end]) / (end - start)

            smoothed.append(BoundingBox(
                x1=int(x1_avg),
                y1=int(y1_avg),
                x2=int(x2_avg),
                y2=int(y2_avg),
                confidence=bboxes[i].confidence
            ))

        return smoothed

    def compute_stable_crop_region(self, bboxes: List[BoundingBox],
                                   img_width: int, img_height: int,
                                   margin: float = 0.3,
                                   target_size: int = 512) -> CropRegion:
        """
        Compute a stable crop region that contains the face throughout the video.

        Args:
            bboxes: List of face bounding boxes for each frame
            img_width: Original image width
            img_height: Original image height
            margin: Extra margin around the face (0.3 = 30%)
            target_size: Target output size (default 512 for GeneFace++)

        Returns:
            CropRegion that stably contains the face
        """
        if not bboxes:
            # Default to center crop
            size = min(img_width, img_height)
            cx, cy = img_width // 2, img_height // 2
            return CropRegion(
                x1=cx - size//2, y1=cy - size//2,
                x2=cx + size//2, y2=cy + size//2,
                original_width=img_width,
                original_height=img_height
            )

        # Find bounding box that contains all face positions
        min_x1 = min(b.x1 for b in bboxes)
        min_y1 = min(b.y1 for b in bboxes)
        max_x2 = max(b.x2 for b in bboxes)
        max_y2 = max(b.y2 for b in bboxes)

        # Add margin
        width = max_x2 - min_x1
        height = max_y2 - min_y1
        margin_w = int(width * margin)
        margin_h = int(height * margin)

        x1 = min_x1 - margin_w
        y1 = min_y1 - margin_h
        x2 = max_x2 + margin_w
        y2 = max_y2 + margin_h

        # Make it square (required for GeneFace++)
        crop_width = x2 - x1
        crop_height = y2 - y1
        crop_size = max(crop_width, crop_height)

        # Center the square
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half = crop_size // 2

        x1 = cx - half
        y1 = cy - half
        x2 = cx + half
        y2 = cy + half

        # Clip to image boundaries and adjust to maintain square
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > img_width:
            x1 -= (x2 - img_width)
            x2 = img_width
        if y2 > img_height:
            y1 -= (y2 - img_height)
            y2 = img_height

        # Final clip
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)

        # Ensure square by taking minimum dimension
        final_width = x2 - x1
        final_height = y2 - y1
        final_size = min(final_width, final_height)

        # Re-center
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half = final_size // 2

        return CropRegion(
            x1=max(0, cx - half),
            y1=max(0, cy - half),
            x2=min(img_width, cx + half),
            y2=min(img_height, cy + half),
            original_width=img_width,
            original_height=img_height
        )

    def __del__(self):
        """Clean up detector resources"""
        if hasattr(self, 'detector'):
            self.detector.close()


def extract_face_crops(frames: np.ndarray,
                       crop_region: CropRegion,
                       target_size: int = 512) -> np.ndarray:
    """
    Extract face crops from video frames.

    Args:
        frames: Video frames [T, H, W, 3]
        crop_region: Stable crop region
        target_size: Output size (default 512)

    Returns:
        Cropped and resized frames [T, target_size, target_size, 3]
    """
    crops = []

    for frame in frames:
        # Extract crop
        crop = frame[crop_region.y1:crop_region.y2,
                     crop_region.x1:crop_region.x2]

        # Resize to target
        if crop.shape[0] != target_size or crop.shape[1] != target_size:
            crop = cv2.resize(crop, (target_size, target_size),
                            interpolation=cv2.INTER_LANCZOS4)

        crops.append(crop)

    return np.stack(crops)


def save_crop_metadata(video_name: str, crop_region: CropRegion,
                       face_bboxes: List[BoundingBox], output_dir: str = None):
    """
    Save crop metadata for later compositing.

    Args:
        video_name: Source video path
        crop_region: Computed crop region
        face_bboxes: Per-frame face bounding boxes
        output_dir: Output directory (default: same as processed video)
    """
    import pickle

    if output_dir is None:
        output_dir = video_name.replace("/raw/", "/processed/").replace(".mp4", "/")

    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        'crop_region': {
            'x1': crop_region.x1,
            'y1': crop_region.y1,
            'x2': crop_region.x2,
            'y2': crop_region.y2,
            'original_width': crop_region.original_width,
            'original_height': crop_region.original_height,
        },
        'face_bboxes': [
            {
                'x1': b.x1, 'y1': b.y1, 'x2': b.x2, 'y2': b.y2,
                'confidence': b.confidence
            } for b in face_bboxes
        ]
    }

    output_path = os.path.join(output_dir, 'crop_metadata.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"Saved crop metadata to {output_path}")


def load_crop_metadata(video_name: str) -> Tuple[CropRegion, List[BoundingBox]]:
    """
    Load previously saved crop metadata.

    Args:
        video_name: Source video path

    Returns:
        Tuple of (CropRegion, List[BoundingBox])
    """
    import pickle

    metadata_path = video_name.replace("/raw/", "/processed/").replace(".mp4", "/crop_metadata.pkl")

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    crop_region = CropRegion(**metadata['crop_region'])
    face_bboxes = [BoundingBox(**b) for b in metadata['face_bboxes']]

    return crop_region, face_bboxes


if __name__ == '__main__':
    # Test the face detector
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default="face_crops.mp4", help="Output video path")
    args = parser.parse_args()

    # Load video
    cap = cv2.VideoCapture(args.video)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    frames = np.array(frames)

    print(f"Loaded {len(frames)} frames of shape {frames.shape}")

    # Detect and track faces
    detector = FaceDetector()
    bboxes = detector.track_faces(frames)

    # Compute stable crop
    crop_region = detector.compute_stable_crop_region(
        bboxes, frames.shape[2], frames.shape[1]
    )
    print(f"Crop region: ({crop_region.x1}, {crop_region.y1}) - ({crop_region.x2}, {crop_region.y2})")

    # Extract crops
    crops = extract_face_crops(frames, crop_region, target_size=512)
    print(f"Extracted crops shape: {crops.shape}")

    # Save output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, 25.0, (512, 512))
    for crop in crops:
        out.write(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    out.release()

    print(f"Saved face crops to {args.output}")
