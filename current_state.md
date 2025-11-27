# Current State: Realtime Face Generation & Lipsync

## Executive Summary

The BeaverVision-2 codebase is built on **GeneFace++**, a neural radiance field (NeRF) based talking face generation system. The current architecture has strict requirements on input video format, particularly around face size and positioning, which limits its applicability to arbitrary videos with human faces.

---

## Architecture Overview

### Three-Stage Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                         INFERENCE PIPELINE                            │
└──────────────────────────────────────────────────────────────────────┘

Audio Input (wav/mp3/mp4)
    │
    ▼
[1] Audio-to-SECC (audio2motion)
    ├── HuBERT Features [1024-dim]
    ├── F0 Pitch Features
    │       ▼
    ├── VAEModel / PitchContourVAEModel
    │       ▼
    └── Identity [80] + Expression [64] coefficients

    │
    ▼
[2] PostNet Refinement (optional)
    ├── CNN-based landmark refinement
    ├── LLE (Local Linear Embedding) projection
    │       ▼
    └── Refined 3D Landmarks [68×3]

    │
    ▼
[3] SECC-to-Video (RADNeRF)
    ├── Neural Radiance Field rendering
    │       ▼
    └── RGB Frames [512×512×3] @ 25 FPS
```

---

## Key Components & Files

| Component | File Location | Purpose |
|-----------|---------------|---------|
| Main Inference | `genefacepp_infer.py` | `GeneFace2Infer` class - orchestrates pipeline |
| Audio2Motion | `modules/audio2motion/vae.py` | VAE model for audio→expression |
| PostNet | `modules/postnet/models.py` | CNN refinement of landmarks |
| RADNeRF | `modules/radnerfs/radnerf.py` | Neural rendering |
| Face Landmarker | `data_gen/utils/mp_feature_extractors/face_landmarker.py` | MediaPipe 478-point detection |
| 3D Face Model | `deep_3drecon/deep_3drecon_models/bfm.py` | Basel Face Model (BFM) |
| 3DMM Fitting | `data_gen/utils/process_video/fit_3dmm_landmark.py` | Fit 3D morphable model |
| Dataset Utils | `tasks/radnerfs/dataset_utils.py` | Training data handling |
| Video Preprocessing | `data_gen/utils/process_video/resample_video_to_25fps_resize_to_512.py` | Video format conversion |

---

## Current Constraints & Limitations

### 1. Video Resolution - FIXED at 512×512

**Location**: Multiple files enforce this constraint

```python
# fit_3dmm_landmark.py:156
assert img_h == img_w  # Forces square video

# resample_video_to_25fps_resize_to_512.py:19-20
assert vid_info['width'] == vid_info['height']
cmd = f'... scale=w=512:h=512 ...'

# genefacepp_infer.py:467
pred_rgb = model_out['rgb_map'][0].reshape([512,512,3])
```

**Impact**: Only square videos are accepted. Non-square videos must be manually cropped.

### 2. Face Must Dominate the Frame

**Evidence from README.md:59**:
> "Please make sure that the head segment occupies a relatively large region in the video (e.g., similar to the provided May.mp4). Or you need to hand-crop your training video."

**Why this constraint exists**:
1. The NeRF is trained on a bounded 3D scene with `bound=1` (coordinates in [-1,1]³)
2. The camera focal length is fixed at `focal=1015` for 224×224 training (scaled to 512)
3. The 3DMM fitting expects face landmarks to occupy a significant portion of the image
4. The face boundary mask generation assumes face-centric framing

### 3. Single Face Detection Only

**Location**: `face_landmarker.py:55-60`
```python
self.image_mode_options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1  # Only one face supported
)
```

### 4. Face Must Be Present in Every Frame

**From fit_3dmm_landmark.py:176-177**:
```python
if lms is None:
    print(f"get None lms_2d, please check whether each frame has one head, exiting...")
    return False
```

If MediaPipe fails to detect a face in any frame, the entire preprocessing fails.

### 5. No Support for Body Parts (Hands, Torso in Main Frame)

The current segmentation model (`MediapipeSegmenter`) separates:
- Background (class 0)
- Head (class 1)
- Neck (class 2)
- Hair (class 3)
- Torso (class 4)
- Body skin (class 5)

However, **hands and gestures are NOT explicitly handled**. The system assumes:
- The torso region is static and can be inpainted
- Body parts don't occlude the face region
- The face boundary mask (convex hull) doesn't intersect with hands

---

## Face Detection & Tracking

### MediaPipe Face Landmarker

- Extracts **478 3D keypoints** per frame
- Operates in two modes combined:
  - **IMAGE mode**: Better for mouth/eye precision (single-frame)
  - **VIDEO mode**: Better for head pose (temporal smoothing)

**Key landmark subsets**:
- `lm68_from_lm478`: 68-point face alignment (used for lipsync)
- `lm131_from_lm478`: Extended contour landmarks
- `lm468_from_lm478`: Full face mesh

### 3DMM Coefficient Fitting

**fit_3dmm_landmark.py** performs iterative optimization:

```python
# Optimized parameters:
id_para     # [80] identity basis coefficients
exp_para    # [64] expression basis coefficients
euler_angle # [3] head rotation (pitch, yaw, roll)
trans       # [3] translation

# Loss functions:
loss_lan   # Landmark reprojection error (weighted)
loss_lap   # Laplacian smoothness
loss_regid # Identity regularization
loss_regexp # Expression regularization
```

**Weighted landmark importance** (`cal_lan_loss_mp`):
- Eyes: weight 3, upper eyelid: weight 20
- Inner/outer lips: weight 5
- Unmatch mask (face boundary): weight 0

---

## Training Data Requirements

### Preprocessing Steps (from docs)

1. **Crop to 512×512, 25 FPS**:
   ```bash
   ffmpeg -i input.mp4 -vf fps=25,scale=w=512:h=512 output.mp4
   ```

2. **Extract ground truth images**:
   ```bash
   ffmpeg ... -start_number 0 .../gt_imgs/%08d.jpg
   ```

3. **Extract 2D landmarks** (`extract_lm2d.py`)

4. **Fit 3DMM coefficients** (`fit_3dmm_landmark.py`)

5. **Extract segments** (head, torso, background)

6. **Train RADNeRF** on the processed data

---

## NeRF Configuration

### Key Parameters (from `egs/datasets/videos/*/radnerf_torso.yaml`)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `H, W` | 512 | Output resolution |
| `bound` | 1 | Scene bounding box [-1,1]³ |
| `camera_scale` | 4.0 | Maps camera to scene bounds |
| `near` | 0.05 | Near clipping plane |
| `far` | 1.0 | Far clipping plane |
| `n_rays` | 65536 | Rays per training iteration |
| `grid_size` | 128 | Density grid resolution |

### Camera Model

- **Pinhole camera** with fixed intrinsics
- `focal=1015` (based on 224×224 training resolution)
- `center=112` (scaled to 256 for 512×512)
- Perspective projection assumed

---

## Inference Flow

### `GeneFace2Infer.forward_system()` Pipeline

1. **`prepare_batch_from_inp()`**:
   - Load audio → extract 16kHz WAV
   - Extract HuBERT features (1024-dim)
   - Extract F0 pitch
   - Load camera poses from trained model

2. **`forward_audio2secc()`**:
   - Audio → Identity/Expression via VAE
   - Optional PostNet refinement
   - LLE blending with training data (default `lle_percent=0.2`)
   - Inject periodic eye blinks (every 100 frames)

3. **`forward_secc2video()`**:
   - For each frame: render via RADNeRF
   - Output fixed 512×512 @ 25 FPS
   - Combine with audio via ffmpeg

---

## Current Pain Points for Arbitrary Videos

| Issue | Root Cause | Impact |
|-------|------------|--------|
| Square format only | Hardcoded 512×512 assertion | Non-square videos fail |
| Large face required | NeRF trained on face-centric data | Small faces produce artifacts |
| No hand/gesture support | Segmentation doesn't handle hands | Hands corrupt face region |
| Static torso assumed | Torso inpainting for NeRF training | Moving gestures break model |
| Single face only | `num_faces=1` in MediaPipe | Multi-person videos fail |
| Face in every frame | Landmark extraction fails otherwise | Occlusions break pipeline |

---

## Summary

The current GeneFace++ implementation is optimized for **controlled, face-centric talking head videos** similar to the provided `May.mp4` sample. It makes strong assumptions about:

1. Video being 512×512 square
2. Face occupying the majority of the frame
3. Minimal body movement (static torso)
4. No hand gestures occluding the face
5. Face visible in every frame

To support arbitrary videos with hand gestures and varying face sizes, significant architectural changes are needed to the preprocessing pipeline, face detection/tracking, and potentially the NeRF rendering approach itself.
