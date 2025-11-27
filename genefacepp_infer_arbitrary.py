"""
GeneFace++ Inference for Arbitrary Videos

This module extends the standard GeneFace++ inference to support:
- Videos of any aspect ratio (not just 512x512)
- Videos where face is not the dominant element
- Videos with hand gestures and body movement
- Compositing lipsynced face back into original video

Usage:
    python genefacepp_infer_arbitrary.py \
        --src_video data/raw/videos/my_video.mp4 \
        --drv_aud data/raw/val_wavs/speech.wav \
        --head_ckpt checkpoints/my_avatar/head \
        --torso_ckpt checkpoints/my_avatar/torso \
        --out_name output/result.mp4
"""

import os
import sys
sys.path.append('./')

import numpy as np
import cv2
import torch
import argparse
import tqdm
from typing import Optional, Dict, Tuple

from genefacepp_infer import GeneFace2Infer
from data_gen.utils.process_video.face_detector import (
    FaceDetector, BoundingBox, CropRegion,
    extract_face_crops, save_crop_metadata, load_crop_metadata
)
from data_gen.utils.process_video.crop_utils import (
    read_video_frames, write_video_frames, VideoInfo,
    composite_face_to_original, create_feather_mask
)


class GeneFace2InferArbitrary(GeneFace2Infer):
    """
    Extended GeneFace++ inference that supports arbitrary video inputs.

    This class adds:
    1. Automatic face detection and cropping for non-square videos
    2. Compositing of generated face back into original video
    3. Support for videos where face is not dominant
    """

    def __init__(self, audio2secc_dir, postnet_dir, head_model_dir, torso_model_dir,
                 device=None, face_margin: float = 0.3):
        """
        Initialize the arbitrary video inference engine.

        Args:
            audio2secc_dir: Path to audio2motion model checkpoint
            postnet_dir: Path to postnet model checkpoint (or '')
            head_model_dir: Path to head RADNeRF model checkpoint
            torso_model_dir: Path to torso RADNeRF model checkpoint (or '')
            device: Device to run on (default: auto-detect)
            face_margin: Margin around detected face for cropping (default 0.3)
        """
        super().__init__(audio2secc_dir, postnet_dir, head_model_dir, torso_model_dir, device)
        self.face_detector = FaceDetector()
        self.face_margin = face_margin

    def infer_arbitrary_video(self, inp: Dict) -> str:
        """
        Perform lipsync inference on an arbitrary video.

        Args:
            inp: Dictionary containing:
                - src_video: Source video path (any aspect ratio)
                - drv_audio_name: Driving audio path
                - out_name: Output video path
                - composite_mode: 'full' (composite back) or 'face_only' (512x512 output)
                - blend_mode: 'direct', 'feather', or 'seamless'
                - Other standard GeneFace++ parameters

        Returns:
            Output video path
        """
        src_video = inp.get('src_video', inp.get('drv_audio_name', '').replace('.wav', '.mp4'))
        composite_mode = inp.get('composite_mode', 'full')
        blend_mode = inp.get('blend_mode', 'feather')

        print(f"Processing arbitrary video: {src_video}")

        # Step 1: Load and analyze source video
        frames, video_info = read_video_frames(src_video)
        print(f"Input video: {video_info.width}x{video_info.height} @ {video_info.fps}fps, {video_info.total_frames} frames")

        # Step 2: Detect and track face
        print("Detecting and tracking face...")
        face_bboxes = self.face_detector.track_faces(frames)
        crop_region = self.face_detector.compute_stable_crop_region(
            face_bboxes, video_info.width, video_info.height,
            margin=self.face_margin
        )
        print(f"Face crop region: ({crop_region.x1}, {crop_region.y1}) - ({crop_region.x2}, {crop_region.y2})")

        # Step 3: Extract face crops for GeneFace++ processing
        face_crops = extract_face_crops(frames, crop_region, target_size=512)
        print(f"Extracted {len(face_crops)} face crops at 512x512")

        # Step 4: Save temporary cropped video for GeneFace++ processing
        temp_crop_video = inp['out_name'].replace('.mp4', '_temp_crop.mp4')
        write_video_frames(face_crops, temp_crop_video, fps=25.0)

        # Step 5: Run standard GeneFace++ inference on cropped video
        # Note: GeneFace++ generates from audio, not from video frames directly
        # The cropped video is used for avatar training, not inference
        # For inference, we just use the trained model
        temp_output = inp['out_name'].replace('.mp4', '_temp_face.mp4')
        face_inp = inp.copy()
        face_inp['out_name'] = temp_output

        print("Running GeneFace++ lipsync inference...")
        self.infer_once(face_inp)

        # Step 6: Load generated face video
        generated_frames, _ = read_video_frames(temp_output)
        print(f"Generated {len(generated_frames)} lipsynced frames")

        if composite_mode == 'face_only':
            # Just rename temp output to final output
            os.rename(temp_output, inp['out_name'])
            # Clean up
            if os.path.exists(temp_crop_video):
                os.remove(temp_crop_video)
            return inp['out_name']

        # Step 7: Composite generated face back into original video
        print(f"Compositing with {blend_mode} blending...")
        composited_frames = []

        n_frames = min(len(frames), len(generated_frames))
        crop_dict = {
            'x1': crop_region.x1,
            'y1': crop_region.y1,
            'x2': crop_region.x2,
            'y2': crop_region.y2
        }

        for i in tqdm.trange(n_frames, desc="Compositing frames"):
            composited = composite_face_to_original(
                frames[i],
                generated_frames[i],
                crop_dict,
                blend_mode=blend_mode
            )
            composited_frames.append(composited)

        composited_frames = np.array(composited_frames)

        # Step 8: Write final output with audio
        temp_composite = inp['out_name'].replace('.mp4', '_temp_composite.mp4')
        write_video_frames(composited_frames, temp_composite, fps=video_info.fps)

        # Add audio
        cmd = f"ffmpeg -i {temp_composite} -i {self.wav16k_name} -y -shortest -c:v libx264 -pix_fmt yuv420p -b:v 2000k -v quiet -shortest {inp['out_name']}"
        ret = os.system(cmd)

        if ret == 0:
            print(f"Saved final output to {inp['out_name']}")
            # Clean up temp files
            for temp_file in [temp_crop_video, temp_output, temp_composite, self.wav16k_name]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        else:
            print(f"Warning: ffmpeg failed with code {ret}")
            # Keep temp files for debugging
            os.rename(temp_composite, inp['out_name'])

        return inp['out_name']

    @classmethod
    def example_run_arbitrary(cls, inp=None):
        """Run example inference on arbitrary video"""
        inp_tmp = {
            'src_video': 'data/raw/videos/example.mp4',
            'drv_audio_name': 'data/raw/val_wavs/zozo.wav',
            'composite_mode': 'full',
            'blend_mode': 'feather',
            'mouth_amp': 0.4,
            'temperature': 0.2,
            'lle_percent': 0.2,
            'blink_mode': 'period',
            'drv_pose': 'static',
            'debug': False,
            'out_name': 'output/result.mp4',
            'raymarching_end_threshold': 0.01,
            'low_memory_usage': False,
        }
        if inp is not None:
            inp_tmp.update(inp)
        inp = inp_tmp

        infer_instance = cls(
            inp['a2m_ckpt'],
            inp['postnet_ckpt'],
            inp['head_ckpt'],
            inp['torso_ckpt'],
            face_margin=inp.get('face_margin', 0.3)
        )

        if 'src_video' in inp and inp['src_video']:
            infer_instance.infer_arbitrary_video(inp)
        else:
            infer_instance.infer_once(inp)


def preprocess_arbitrary_video(video_path: str,
                                output_dir: str,
                                face_margin: float = 0.3) -> Dict:
    """
    Preprocess an arbitrary video for GeneFace++ avatar training.

    This function:
    1. Detects and tracks face in video
    2. Extracts face-centric crops at 512x512
    3. Saves preprocessed video and metadata

    Args:
        video_path: Input video path
        output_dir: Output directory
        face_margin: Margin around detected face

    Returns:
        Dictionary with preprocessing metadata
    """
    print(f"Preprocessing arbitrary video: {video_path}")

    # Load video
    frames, video_info = read_video_frames(video_path)
    print(f"Input: {video_info.width}x{video_info.height} @ {video_info.fps}fps")

    # Detect and track face
    detector = FaceDetector()
    face_bboxes = detector.track_faces(frames)
    crop_region = detector.compute_stable_crop_region(
        face_bboxes, video_info.width, video_info.height,
        margin=face_margin
    )

    print(f"Face region: ({crop_region.x1}, {crop_region.y1}) - ({crop_region.x2}, {crop_region.y2})")

    # Extract face crops
    face_crops = extract_face_crops(frames, crop_region, target_size=512)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.basename(video_path).replace('.mp4', '')
    output_video = os.path.join(output_dir, f"{video_name}_512.mp4")
    write_video_frames(face_crops, output_video, fps=25.0)
    print(f"Saved preprocessed video: {output_video}")

    # Save metadata
    metadata = {
        'original_video': video_path,
        'original_width': video_info.width,
        'original_height': video_info.height,
        'original_fps': video_info.fps,
        'crop_region': {
            'x1': crop_region.x1,
            'y1': crop_region.y1,
            'x2': crop_region.x2,
            'y2': crop_region.y2,
        },
        'face_bboxes': [
            {'x1': b.x1, 'y1': b.y1, 'x2': b.x2, 'y2': b.y2, 'confidence': b.confidence}
            for b in face_bboxes
        ],
        'output_video': output_video,
    }

    metadata_path = os.path.join(output_dir, f"{video_name}_crop_metadata.npy")
    np.save(metadata_path, metadata, allow_pickle=True)
    print(f"Saved metadata: {metadata_path}")

    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GeneFace++ Inference for Arbitrary Videos")

    # Mode selection
    parser.add_argument("--mode", choices=['infer', 'preprocess'], default='infer',
                       help="'infer' for lipsync inference, 'preprocess' for avatar training prep")

    # Input/output
    parser.add_argument("--src_video", type=str, default='',
                       help="Source video path (for arbitrary video mode)")
    parser.add_argument("--drv_aud", type=str, default='data/raw/val_wavs/MacronSpeech.wav',
                       help="Driving audio path")
    parser.add_argument("--out_name", type=str, default='output/result.mp4',
                       help="Output video path")
    parser.add_argument("--output_dir", type=str, default='data/processed/videos/',
                       help="Output directory for preprocessing mode")

    # Model checkpoints
    parser.add_argument("--a2m_ckpt", default='checkpoints/audio2motion_vae',
                       help="Audio2motion model checkpoint")
    parser.add_argument("--postnet_ckpt", default='',
                       help="PostNet model checkpoint")
    parser.add_argument("--head_ckpt", default='',
                       help="Head RADNeRF model checkpoint")
    parser.add_argument("--torso_ckpt", default='',
                       help="Torso RADNeRF model checkpoint")

    # Compositing options
    parser.add_argument("--composite_mode", choices=['full', 'face_only'], default='full',
                       help="'full' composites face back into original, 'face_only' outputs 512x512")
    parser.add_argument("--blend_mode", choices=['direct', 'feather', 'seamless'], default='feather',
                       help="Blending mode for compositing")
    parser.add_argument("--face_margin", type=float, default=0.3,
                       help="Margin around detected face (0.0-1.0)")

    # GeneFace++ parameters
    parser.add_argument("--drv_pose", default='static',
                       help="Pose mode: 'static' or 'start-end' range")
    parser.add_argument("--blink_mode", default='period',
                       help="Blink mode: 'none' or 'period'")
    parser.add_argument("--lle_percent", type=float, default=0.2,
                       help="LLE blending percent")
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="VAE sampling temperature")
    parser.add_argument("--mouth_amp", type=float, default=0.4,
                       help="Mouth amplitude")
    parser.add_argument("--raymarching_end_threshold", type=float, default=0.01,
                       help="Raymarching threshold (higher = faster)")
    parser.add_argument("--debug", action='store_true',
                       help="Enable debug mode")
    parser.add_argument("--low_memory_usage", action='store_true',
                       help="Use low memory mode")

    args = parser.parse_args()

    if args.mode == 'preprocess':
        if not args.src_video:
            raise ValueError("--src_video required for preprocess mode")
        preprocess_arbitrary_video(
            args.src_video,
            args.output_dir,
            face_margin=args.face_margin
        )
    else:  # infer mode
        inp = {
            'a2m_ckpt': args.a2m_ckpt,
            'postnet_ckpt': args.postnet_ckpt,
            'head_ckpt': args.head_ckpt,
            'torso_ckpt': args.torso_ckpt,
            'src_video': args.src_video,
            'drv_audio_name': args.drv_aud,
            'drv_pose': args.drv_pose,
            'blink_mode': args.blink_mode,
            'temperature': args.temperature,
            'mouth_amp': args.mouth_amp,
            'lle_percent': args.lle_percent,
            'debug': args.debug,
            'out_name': args.out_name,
            'raymarching_end_threshold': args.raymarching_end_threshold,
            'low_memory_usage': args.low_memory_usage,
            'composite_mode': args.composite_mode,
            'blend_mode': args.blend_mode,
            'face_margin': args.face_margin,
        }

        GeneFace2InferArbitrary.example_run_arbitrary(inp)
