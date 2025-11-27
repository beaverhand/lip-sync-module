"""
Crop Utilities for Arbitrary Video Lipsync

This module provides utilities for:
- Dynamic video cropping based on face detection
- Frame extraction with arbitrary aspect ratios
- Compositing lipsynced faces back into original frames
"""

import numpy as np
import cv2
import os
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class VideoInfo:
    """Video metadata"""
    width: int
    height: int
    fps: float
    total_frames: int
    is_square: bool

    @classmethod
    def from_video(cls, video_path: str) -> 'VideoInfo':
        """Extract video info from file"""
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        return cls(
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            is_square=(width == height)
        )


def read_video_frames(video_path: str,
                      max_frames: Optional[int] = None) -> Tuple[np.ndarray, VideoInfo]:
    """
    Read video frames into numpy array.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to read (None = all)

    Returns:
        Tuple of (frames [T, H, W, 3] RGB, VideoInfo)
    """
    info = VideoInfo.from_video(video_path)

    cap = cv2.VideoCapture(video_path)
    frames = []

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

        count += 1
        if max_frames is not None and count >= max_frames:
            break

    cap.release()

    return np.array(frames), info


def write_video_frames(frames: np.ndarray,
                       output_path: str,
                       fps: float = 25.0,
                       codec: str = 'mp4v'):
    """
    Write frames to video file.

    Args:
        frames: Video frames [T, H, W, 3] RGB
        output_path: Output video path
        fps: Frames per second
        codec: Video codec (default 'mp4v')
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    h, w = frames.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()


def crop_and_resize(frame: np.ndarray,
                    x1: int, y1: int, x2: int, y2: int,
                    target_size: int = 512) -> np.ndarray:
    """
    Crop frame and resize to target size.

    Args:
        frame: Input frame [H, W, 3]
        x1, y1, x2, y2: Crop coordinates
        target_size: Output size (square)

    Returns:
        Cropped and resized frame [target_size, target_size, 3]
    """
    crop = frame[y1:y2, x1:x2]

    if crop.shape[0] != target_size or crop.shape[1] != target_size:
        crop = cv2.resize(crop, (target_size, target_size),
                         interpolation=cv2.INTER_LANCZOS4)

    return crop


def pad_to_square(frame: np.ndarray,
                  pad_value: int = 0) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Pad frame to make it square.

    Args:
        frame: Input frame [H, W, 3]
        pad_value: Padding value (default black)

    Returns:
        Tuple of (padded frame, (top, bottom, left, right) padding)
    """
    h, w = frame.shape[:2]

    if h == w:
        return frame, (0, 0, 0, 0)

    size = max(h, w)

    # Calculate padding
    pad_h = size - h
    pad_w = size - w

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # Apply padding
    padded = cv2.copyMakeBorder(
        frame, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[pad_value] * 3
    )

    return padded, (top, bottom, left, right)


def unpad_frame(frame: np.ndarray,
                padding: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Remove padding from frame.

    Args:
        frame: Padded frame [H, W, 3]
        padding: (top, bottom, left, right) padding values

    Returns:
        Unpadded frame
    """
    top, bottom, left, right = padding
    h, w = frame.shape[:2]

    y1 = top
    y2 = h - bottom if bottom > 0 else h
    x1 = left
    x2 = w - right if right > 0 else w

    return frame[y1:y2, x1:x2]


def preprocess_video_for_geneface(video_path: str,
                                   output_path: str,
                                   target_size: int = 512,
                                   target_fps: float = 25.0,
                                   use_face_detection: bool = True,
                                   margin: float = 0.3) -> dict:
    """
    Preprocess any video for GeneFace++ pipeline.

    This function handles:
    - Non-square videos
    - Videos where face is not dominant
    - Arbitrary resolutions

    Args:
        video_path: Input video path
        output_path: Output video path (512x512, 25fps)
        target_size: Output size (default 512)
        target_fps: Output FPS (default 25)
        use_face_detection: Whether to use face detection for cropping
        margin: Margin around detected face (default 0.3)

    Returns:
        Dictionary with crop metadata
    """
    from .face_detector import FaceDetector, extract_face_crops, save_crop_metadata

    # Read video
    frames, info = read_video_frames(video_path)
    print(f"Input video: {info.width}x{info.height} @ {info.fps}fps, {info.total_frames} frames")

    metadata = {
        'original_width': info.width,
        'original_height': info.height,
        'original_fps': info.fps,
        'target_size': target_size,
        'use_face_detection': use_face_detection,
    }

    if use_face_detection:
        # Detect and track face
        detector = FaceDetector()
        bboxes = detector.track_faces(frames)

        # Compute stable crop region
        crop_region = detector.compute_stable_crop_region(
            bboxes, info.width, info.height, margin=margin
        )

        print(f"Face crop region: ({crop_region.x1}, {crop_region.y1}) - ({crop_region.x2}, {crop_region.y2})")

        # Extract crops
        processed_frames = extract_face_crops(frames, crop_region, target_size)

        # Save metadata
        metadata['crop_region'] = {
            'x1': crop_region.x1,
            'y1': crop_region.y1,
            'x2': crop_region.x2,
            'y2': crop_region.y2,
        }
        metadata['face_bboxes'] = [
            {'x1': b.x1, 'y1': b.y1, 'x2': b.x2, 'y2': b.y2, 'confidence': b.confidence}
            for b in bboxes
        ]

    else:
        # Simple center crop
        if info.is_square:
            # Already square, just resize
            processed_frames = np.array([
                cv2.resize(f, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
                for f in frames
            ])
            metadata['crop_region'] = {
                'x1': 0, 'y1': 0,
                'x2': info.width, 'y2': info.height
            }
        else:
            # Center crop to square
            size = min(info.width, info.height)
            x1 = (info.width - size) // 2
            y1 = (info.height - size) // 2
            x2 = x1 + size
            y2 = y1 + size

            processed_frames = np.array([
                crop_and_resize(f, x1, y1, x2, y2, target_size)
                for f in frames
            ])
            metadata['crop_region'] = {
                'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2
            }

    # Write output video
    write_video_frames(processed_frames, output_path, fps=target_fps)
    print(f"Saved preprocessed video to {output_path}")

    # Save metadata
    metadata_path = output_path.replace('.mp4', '_crop_metadata.pkl')
    import pickle
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved crop metadata to {metadata_path}")

    return metadata


def composite_face_to_original(original_frame: np.ndarray,
                                generated_face: np.ndarray,
                                crop_region: dict,
                                blend_mode: str = 'direct') -> np.ndarray:
    """
    Composite generated face back into original frame.

    Args:
        original_frame: Original video frame [H, W, 3]
        generated_face: Generated face from GeneFace++ [512, 512, 3]
        crop_region: Crop region dictionary with x1, y1, x2, y2
        blend_mode: 'direct' for simple paste, 'seamless' for Poisson blending

    Returns:
        Composited frame
    """
    x1, y1, x2, y2 = crop_region['x1'], crop_region['y1'], crop_region['x2'], crop_region['y2']
    crop_width = x2 - x1
    crop_height = y2 - y1

    # Resize generated face to match crop size
    face_resized = cv2.resize(generated_face, (crop_width, crop_height),
                              interpolation=cv2.INTER_LANCZOS4)

    result = original_frame.copy()

    if blend_mode == 'direct':
        # Simple paste
        result[y1:y2, x1:x2] = face_resized

    elif blend_mode == 'seamless':
        # Poisson blending for seamless compositing
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Create mask (full white = blend everything)
        mask = np.ones((crop_height, crop_width), dtype=np.uint8) * 255

        # Need BGR for seamlessClone
        face_bgr = cv2.cvtColor(face_resized.astype(np.uint8), cv2.COLOR_RGB2BGR)
        original_bgr = cv2.cvtColor(original_frame.astype(np.uint8), cv2.COLOR_RGB2BGR)

        result_bgr = cv2.seamlessClone(face_bgr, original_bgr, mask, center, cv2.MIXED_CLONE)
        result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    elif blend_mode == 'feather':
        # Feathered alpha blending
        # Create soft mask with feathered edges
        mask = create_feather_mask(crop_height, crop_width, feather_amount=20)

        # Blend
        mask_3d = np.stack([mask] * 3, axis=-1)
        blended = face_resized * mask_3d + result[y1:y2, x1:x2] * (1 - mask_3d)
        result[y1:y2, x1:x2] = blended.astype(np.uint8)

    return result


def create_feather_mask(height: int, width: int, feather_amount: int = 20) -> np.ndarray:
    """
    Create a feathered mask for smooth blending.

    Args:
        height: Mask height
        width: Mask width
        feather_amount: Pixels to feather at edges

    Returns:
        Mask with values 0-1, feathered at edges
    """
    mask = np.ones((height, width), dtype=np.float32)

    # Create distance from edge
    for i in range(feather_amount):
        alpha = i / feather_amount
        mask[i, :] *= alpha
        mask[height - 1 - i, :] *= alpha
        mask[:, i] *= alpha
        mask[:, width - 1 - i] *= alpha

    return mask


def composite_video(original_video_path: str,
                    generated_video_path: str,
                    output_path: str,
                    crop_metadata_path: str = None,
                    blend_mode: str = 'feather') -> None:
    """
    Composite generated lipsync video back into original video.

    Args:
        original_video_path: Path to original video
        generated_video_path: Path to generated 512x512 video from GeneFace++
        output_path: Path to save composited video
        crop_metadata_path: Path to crop metadata (auto-detected if None)
        blend_mode: Blending mode ('direct', 'seamless', 'feather')
    """
    import pickle

    # Load metadata
    if crop_metadata_path is None:
        crop_metadata_path = generated_video_path.replace('.mp4', '_crop_metadata.pkl')
        if not os.path.exists(crop_metadata_path):
            # Try original video path
            crop_metadata_path = original_video_path.replace('.mp4', '_crop_metadata.pkl')

    with open(crop_metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    crop_region = metadata['crop_region']

    # Read videos
    original_frames, orig_info = read_video_frames(original_video_path)
    generated_frames, gen_info = read_video_frames(generated_video_path)

    print(f"Original: {len(original_frames)} frames, Generated: {len(generated_frames)} frames")

    # Composite each frame
    composited_frames = []
    n_frames = min(len(original_frames), len(generated_frames))

    for i in range(n_frames):
        composited = composite_face_to_original(
            original_frames[i],
            generated_frames[i],
            crop_region,
            blend_mode=blend_mode
        )
        composited_frames.append(composited)

    composited_frames = np.array(composited_frames)

    # Write output
    write_video_frames(composited_frames, output_path, fps=orig_info.fps)
    print(f"Saved composited video to {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['preprocess', 'composite'], required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--generated", type=str, help="Generated video for composite mode")
    parser.add_argument("--use-face-detection", action='store_true', default=True)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--blend-mode", choices=['direct', 'seamless', 'feather'], default='feather')

    args = parser.parse_args()

    if args.mode == 'preprocess':
        preprocess_video_for_geneface(
            args.input, args.output,
            use_face_detection=args.use_face_detection,
            margin=args.margin
        )
    elif args.mode == 'composite':
        if not args.generated:
            raise ValueError("--generated required for composite mode")
        composite_video(
            args.input, args.generated, args.output,
            blend_mode=args.blend_mode
        )
