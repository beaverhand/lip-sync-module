# Plan & Implications: Lipsync for Any Video with Human Face

## Goal

Enable lipsyncing for **any video containing a human face** (single actor), including:
- Videos with varying aspect ratios (not just 512×512)
- Videos where the face is a smaller portion of the frame
- Videos with hand gestures and body movement
- Videos with occasional face occlusions

---

## Proposed Architecture Changes

### High-Level Strategy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     NEW PIPELINE ARCHITECTURE                            │
└─────────────────────────────────────────────────────────────────────────┘

Input Video (any resolution/aspect ratio)
    │
    ▼
[NEW] Face Detection & ROI Extraction
    ├── Detect face bounding box per frame
    ├── Track face across frames (temporal consistency)
    ├── Compute dynamic crop region (with margin)
    └── Handle occlusions (hand gestures, etc.)
    │
    ▼
[MODIFIED] Face-Centric Processing
    ├── Extract face region at 512×512
    ├── Run existing GeneFace++ pipeline
    └── Generate lipsynced face
    │
    ▼
[NEW] Face Compositing
    ├── Blend lipsynced face back into original frame
    ├── Handle boundary seamlessly
    └── Preserve hand gestures and body
    │
    ▼
Output Video (original resolution preserved)
```

---

## Implementation Plan

### Phase 1: Dynamic Face Detection & Cropping

#### 1.1 Add Face Detection Module

**New file**: `data_gen/utils/process_video/face_detector.py`

```python
# Functionality needed:
class FaceDetector:
    def detect_face_bbox(self, frame) -> BoundingBox:
        """Return face bounding box with confidence score"""

    def track_faces(self, frames) -> List[BoundingBox]:
        """Track face across video with temporal smoothing"""

    def compute_stable_crop_region(self, bboxes, margin=0.3) -> CropRegion:
        """Compute stable crop that contains face throughout video"""
```

**Options for face detection**:
1. **RetinaFace** - High accuracy, handles occlusions well
2. **MTCNN** - Fast, good for video
3. **MediaPipe Face Detection** - Already in codebase, lightweight
4. **InsightFace** - Robust to pose variations

**Implications**:
- Need to add new dependency (recommend RetinaFace or use existing MediaPipe)
- Add face tracking with temporal smoothing to avoid jittery crops
- Handle cases where face is temporarily occluded

#### 1.2 Modify Preprocessing Pipeline

**File to modify**: `data_gen/utils/process_video/fit_3dmm_landmark.py`

**Current constraint (line 156)**:
```python
assert img_h == img_w  # REMOVE THIS
```

**Changes needed**:
1. Remove square video assertion
2. Add dynamic face ROI extraction before landmark fitting
3. Apply 3DMM fitting only to the cropped face region
4. Store crop coordinates for later compositing

```python
# New flow:
def fit_3dmm_for_a_video(video_name, ...):
    frames = read_video_to_frames(video_name)

    # [NEW] Detect and track face
    face_detector = FaceDetector()
    face_bboxes = face_detector.track_faces(frames)
    crop_region = face_detector.compute_stable_crop_region(face_bboxes)

    # [NEW] Extract face-centric crops
    face_crops = extract_face_crops(frames, crop_region, target_size=512)

    # [EXISTING] Run 3DMM fitting on crops
    # ... existing code on face_crops instead of frames ...

    # [NEW] Save crop metadata for compositing
    save_crop_metadata(video_name, crop_region, face_bboxes)
```

---

### Phase 2: Occlusion-Aware Processing

#### 2.1 Hand/Gesture Detection

**New file**: `data_gen/utils/process_video/occlusion_detector.py`

```python
class OcclusionDetector:
    def detect_hands(self, frame) -> List[BoundingBox]:
        """Detect hand regions using MediaPipe Hands"""

    def detect_face_occlusions(self, frame, face_bbox) -> OcclusionMask:
        """Identify parts of face occluded by hands/objects"""

    def generate_occlusion_aware_mask(self, face_landmarks, hand_bboxes) -> Mask:
        """Create mask excluding occluded face regions"""
```

**Options**:
1. **MediaPipe Hands** - Already compatible with existing stack
2. **Hand segmentation via existing MediapipeSegmenter** - Check if body skin class captures hands

**Implications**:
- The current `inpaint_torso_job()` in `extract_segment_imgs.py` assumes static torso
- Need to handle dynamic hand positions overlaying face region
- May need to exclude occluded face regions from 3DMM fitting loss

#### 2.2 Modify Landmark Fitting for Occlusions

**File**: `fit_3dmm_landmark.py`

**Current behavior**: All 478 landmarks weighted equally (with some adjustments)

**Needed changes**:
```python
def cal_lan_loss_mp_with_occlusion(proj_lan, gt_lan, occlusion_mask):
    """
    Modified loss function that ignores occluded landmarks
    """
    loss = (proj_lan - gt_lan).pow(2)

    # Apply occlusion mask - zero out loss for occluded points
    occlusion_weights = 1.0 - occlusion_mask  # 0 where occluded
    loss = loss * occlusion_weights

    # Existing weights...
    weights[:, eye] = 3
    weights[:, inner_lip] = 5
    # ...

    loss = loss * weights * occlusion_weights
    return torch.mean(loss)
```

---

### Phase 3: Modified NeRF Training/Inference

#### 3.1 Face-Only NeRF with Dynamic Background

**Current**: RADNeRF renders full 512×512 frame including torso/background

**Proposed**: Train/infer NeRF only on face region

**Files to modify**:
- `modules/radnerfs/radnerf.py`
- `tasks/radnerfs/dataset_utils.py`

**Changes**:
1. Modify `RADNeRFDataset` to accept variable face regions
2. Train NeRF on face-only crops (256×256 or 512×512)
3. Use face boundary mask more aggressively to limit rendering region

**Implications**:
- Faster training (smaller region to render)
- Need to retrain models on face-centric crops
- Boundary blending becomes critical

#### 3.2 Alternative: Use Pre-trained Face NeRF

For faster deployment, could use a single well-trained face NeRF and apply style transfer:

1. Train universal face NeRF on diverse dataset
2. At inference: adapt to target identity via fine-tuning or conditioning
3. Focus only on lip region modification

---

### Phase 4: Compositing Pipeline

#### 4.1 New Compositing Module

**New file**: `inference/face_compositor.py`

```python
class FaceCompositor:
    def __init__(self, blend_mode='seamless'):
        self.blend_mode = blend_mode

    def composite(self,
                  original_frame: np.ndarray,
                  generated_face: np.ndarray,
                  crop_region: CropRegion,
                  face_mask: np.ndarray) -> np.ndarray:
        """
        Blend generated face back into original frame

        Args:
            original_frame: Full resolution original frame
            generated_face: 512×512 lipsynced face
            crop_region: Where to place the face
            face_mask: Soft mask for blending (handles hands/occlusions)

        Returns:
            Composited frame at original resolution
        """
        # 1. Resize generated face to match crop size
        face_resized = cv2.resize(generated_face, crop_region.size)

        # 2. Create blending mask (soft edges)
        blend_mask = self._create_blend_mask(face_mask, crop_region)

        # 3. Composite using Poisson blending or alpha blending
        if self.blend_mode == 'seamless':
            result = cv2.seamlessClone(face_resized, original_frame,
                                        blend_mask, crop_region.center,
                                        cv2.MIXED_CLONE)
        else:
            result = self._alpha_blend(original_frame, face_resized,
                                        blend_mask, crop_region)

        return result
```

**Blending options**:
1. **Poisson/Seamless Clone** - Best for color matching
2. **Feathered alpha blending** - Simple, fast
3. **Neural blending** - Train small network for seamless transitions

#### 4.2 Handle Hand Gestures

When hands are in frame:
1. Detect hand regions in original frame
2. Exclude hand regions from face mask
3. Composite face BEHIND hands (hands overlay face)

```python
def composite_with_hands(original, generated_face, crop_region, hand_masks):
    # First, composite face
    result = composite(original, generated_face, crop_region)

    # Then, paste original hands back on top
    for hand_mask in hand_masks:
        result = np.where(hand_mask, original, result)

    return result
```

---

### Phase 5: Modified Inference Pipeline

#### 5.1 Update `genefacepp_infer.py`

**New inference flow**:

```python
class GeneFace2InferArbitrary(GeneFace2Infer):
    def __init__(self, ...):
        super().__init__(...)
        self.face_detector = FaceDetector()
        self.occlusion_detector = OcclusionDetector()
        self.compositor = FaceCompositor()

    def infer_once_arbitrary(self, inp):
        # 1. Load source video
        video_frames = load_video(inp['src_video'])
        original_resolution = video_frames[0].shape[:2]

        # 2. Detect and track face
        face_bboxes = self.face_detector.track_faces(video_frames)
        crop_region = self.face_detector.compute_stable_crop_region(face_bboxes)

        # 3. Extract face crops
        face_crops = extract_crops(video_frames, crop_region, size=512)

        # 4. Detect occlusions (hands)
        hand_masks = [self.occlusion_detector.detect_hands(f) for f in video_frames]

        # 5. Run existing lipsync on face crops
        lipsynced_faces = self.run_lipsync(face_crops, inp['audio'])

        # 6. Composite back
        output_frames = []
        for i, (orig, synced) in enumerate(zip(video_frames, lipsynced_faces)):
            composited = self.compositor.composite(
                orig, synced, crop_region,
                hand_masks[i]
            )
            output_frames.append(composited)

        # 7. Write output at original resolution
        write_video(output_frames, inp['out_name'], fps=25)
```

---

## Summary of Files to Modify/Create

### New Files

| File | Purpose |
|------|---------|
| `data_gen/utils/process_video/face_detector.py` | Face detection & tracking |
| `data_gen/utils/process_video/occlusion_detector.py` | Hand/occlusion detection |
| `data_gen/utils/process_video/crop_utils.py` | Dynamic cropping utilities |
| `inference/face_compositor.py` | Blend face back into original |
| `genefacepp_infer_arbitrary.py` | New inference class for arbitrary videos |

### Files to Modify

| File | Changes |
|------|---------|
| `fit_3dmm_landmark.py:156` | Remove `assert img_h == img_w` |
| `fit_3dmm_landmark.py` | Add dynamic crop before fitting |
| `face_landmarker.py` | Handle partial face visibility |
| `extract_segment_imgs.py` | Occlusion-aware segmentation |
| `resample_video_to_25fps_resize_to_512.py` | Preserve aspect ratio option |
| `genefacepp_infer.py` | Add arbitrary video support |
| `tasks/radnerfs/dataset_utils.py` | Variable face region support |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Jittery face crops | Visual artifacts | Temporal smoothing with Kalman filter |
| Hand occlusions break 3DMM | Bad landmarks | Occlusion-aware loss, exclude occluded points |
| Compositing artifacts | Visible seams | Poisson blending, soft masks |
| Small faces lose detail | Poor lipsync quality | Minimum face size threshold, upscaling |
| Performance degradation | Slow inference | GPU-accelerated face detection |
| Retraining needed | Time/compute cost | Fine-tune existing models on diverse crops |

---

## Implementation Priority

### High Priority (Core Functionality)
1. Face detection & stable cropping
2. Remove square video constraint
3. Basic compositing

### Medium Priority (Quality Improvements)
4. Hand/occlusion detection
5. Seamless blending
6. Temporal consistency

### Lower Priority (Nice to Have)
7. Multi-face support
8. Dynamic background handling
9. Real-time processing optimization

---

## Estimated Effort

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1: Dynamic Face Detection | 2-3 days | None |
| Phase 2: Occlusion Handling | 2-3 days | Phase 1 |
| Phase 3: NeRF Modifications | 3-5 days | Phase 1, 2 |
| Phase 4: Compositing | 2-3 days | Phase 1 |
| Phase 5: Integration | 2-3 days | All phases |
| Testing & Refinement | 3-5 days | All phases |

**Total estimated effort**: 2-3 weeks for MVP, additional time for quality refinement.

---

## Alternative Approaches

### Option A: Crop + Pad (Simplest)
- Detect face, crop with large margin
- Pad to 512×512 square
- Run existing pipeline
- Crop result back to original
- **Pros**: Minimal code changes
- **Cons**: Loses background quality

### Option B: Face Swapping Approach
- Use face swapping techniques (SimSwap, etc.)
- Replace only lip region in original video
- **Pros**: Preserves all background/hands
- **Cons**: May have identity artifacts

### Option C: 2D Lipsync (Wav2Lip style)
- Skip 3D NeRF entirely
- Use 2D GAN-based lip sync
- **Pros**: Much simpler, handles any video
- **Cons**: Lower quality than 3D approach

---

## Recommendation

Start with **Phase 1 (Dynamic Face Detection)** as it provides the most value with least risk. This alone would enable:
- Videos with smaller faces
- Non-square aspect ratios
- Face not centered

Then progressively add occlusion handling and compositing as needed.
