# Cosmos Transfer LoRA Fine-Tuning: Data Pipeline for Aigen Agricultural Data

**Version 2.0 — January 2026**

---

## 1. Executive Summary

This document specifies the end-to-end data pipeline for LoRA fine-tuning Cosmos Transfer on agricultural field data captured by Aigen robots. It is grounded in the actual source data available in S3 (`s3://aigen-iot-data-dev/`) and designed for initial training on a DGX Spark.

**Goal:** Adapt Cosmos Transfer to generate realistic agricultural field video (crop rows, under-canopy views, field conditions) using depth-conditioned video generation.

| Aspect | Detail |
|--------|--------|
| **Training type** | LoRA (Low-Rank Adaptation), ~2% of parameters |
| **Target resolution** | 1280x720 (720p) |
| **Target FPS** | 10 FPS |
| **Clip duration** | 10 seconds (100 frames; 93 minimum required by training) |
| **Data volume target** | 1,000-5,000 clips |
| **Training hardware** | DGX Spark (1x GB10, 128 GB shared VRAM) |
| **Source data** | ~30 robots, ~5,000 capture archives in S3 |

---

## 2. Source Data Audit

### 2.1 S3 Bucket Structure

All data lives in `s3://aigen-iot-data-dev/`:

```
s3://aigen-iot-data-dev/
├── data-capture/              # Primary source: robot capture archives
│   ├── gadget-113/            # ~30 robot name prefixes
│   │   ├── 2025-05-07-T18:40:03UTC.tar
│   │   ├── 2025-05-13-T19:28:35UTC.tar
│   │   └── ...                # ~158 captures for this robot
│   ├── demeter-117/
│   ├── gopherus-111/
│   └── ...
├── session/                   # Session telemetry archives (larger, longer runs)
│   ├── gadget-113/
│   │   ├── 2025-04-08-T18:58:07UTC.tar  # Can be multi-GB
│   │   └── ...                # ~500 sessions for this robot
│   └── ...
├── bowles-data/               # Site-specific data (Bowles farm)
└── calibration/               # Per-robot calibration data
```

**Scale:** ~30 robots total across `data-capture/` and `session/`. Approximately 5,000 capture tar files in `data-capture/` alone.

### 2.2 Capture Folder Structure

Each tar archive (e.g., `2025-11-24-T16:47:59UTC.tar`) extracts to:

```
2025-11-24-T16:47:59UTC/
├── capture/                            # High-quality capture recordings
│   ├── capture_color_nav_front_0_*.mp4       # Nav front RGB
│   ├── capture_depth_nav_front_0_*.mp4       # Nav front depth
│   ├── capture_metadata_nav_front_0_*.csv    # Per-frame timestamps
│   ├── capture_color_nav_rear_0_*.mp4        # Nav rear RGB
│   ├── capture_depth_nav_rear_0_*.mp4        # Nav rear depth
│   ├── capture_metadata_nav_rear_0_*.csv
│   ├── capture_color_crop_center_1_*.mp4     # Crop cam 1 RGB
│   ├── capture_depth_crop_center_1_*.mp4     # Crop cam 1 depth
│   ├── capture_metadata_crop_center_1_*.csv
│   ├── capture_color_crop_center_2_*.mp4     # Crop cam 2 RGB
│   ├── capture_depth_crop_center_2_*.mp4     # Crop cam 2 depth
│   └── capture_metadata_crop_center_2_*.csv
├── video/                              # Streaming recordings (lower quality)
│   ├── nav_front_0_color_*.mp4         # 640x480, 15fps, ~24s
│   ├── nav_rear_0_color_*.mp4
│   ├── crop_center_1_color_*.mp4       # 640x480 (downsampled), 15fps
│   └── crop_center_2_color_*.mp4
├── data/                               # Binary telemetry chunks (XYLEM format)
│   └── chunk_000                       # Contains frame transforms, odometry, etc.
├── conf/                               # Robot configuration at capture time
│   ├── camera/
│   │   ├── nav.yml                     # Nav camera config
│   │   ├── crop.yml                    # Crop camera config
│   │   └── common.yml                  # Shared camera config
│   └── nodes/
│       └── calibration.yml             # Payload mode definitions
└── deployment.txt                      # Software version (e.g., "Aigen Rhizome MAIN v2.0.372")
```

### 2.3 Camera Specifications

| Camera | Capture Resolution | Capture FPS | Streaming Resolution | Max Depth | Aspect Ratio |
|--------|-------------------|-------------|---------------------|-----------|--------------|
| `nav_front_0` | 640x480 | 5 | 640x480 @ 15fps | 2550 mm | 4:3 |
| `nav_rear_0` | 640x480 | 5 | 640x480 @ 15fps | 2550 mm | 4:3 |
| `crop_center_1` | 1280x720 | 5 | 640x480 @ 15fps | 1020 mm | 16:9 |
| `crop_center_2` | 1280x720 | 5 | 640x480 @ 15fps | 1020 mm | 16:9 |

**Key observations:**
- **Crop cameras capture at native 1280x720** — already the target resolution for Cosmos 720p.
- Nav cameras capture at only 640x480 (4:3) — requires upscaling and padding/cropping to reach 1280x720 (16:9).
- Both `capture/` and `video/` have paired depth videos alongside color.
- Capture recordings are variable length (~13-15s observed), not constant.
- Metadata CSVs provide per-frame Unix timestamps with nanosecond precision for both color and depth.

**Depth encoding:** Capture depth videos use 8-bit quantization:
- `depth_mm = (encoded_byte * max_depth_mm) / 255`
- Nav cameras: 10.0 mm per bit (max 2550 mm)
- Crop cameras: 4.0 mm per bit (max 1020 mm)

### 2.4 Camera Orientation

Each robot has fixed camera mounting positions:

| Camera | Orientation | View Type | How We Know |
|--------|-------------|-----------|-------------|
| `nav_front_0` | Forward-facing, ~0 pitch | Row corridor view | Camera type + name |
| `nav_rear_0` | Rear-facing, ~0 pitch | Row corridor (behind) | Camera type + name |
| `crop_center_1` | Payload-mounted, variable | Top-down or angled | Telemetry extrinsics (see 4.3) |
| `crop_center_2` | Payload-mounted, variable | Top-down or angled | Telemetry extrinsics (see 4.3) |

The `calibration.yml` config defines two known payload modes and a mapping from chassis types to default orientations:
- `payload_td` (type 0): **top_down** — camera looking straight down at canopy
- `payload_ir` (type 1): **intra_row** — camera at an angle for under-canopy/between-row view
- `camera_spacing` maps chassis type → orientation: `trex` → `762_ir` (intra-row), `2x` → `838_td` (top-down), `3x` → `762_td` (top-down)

However, there is no straightforward config field that indicates which chassis type or mode is active for a given capture. The definitive approach is to extract camera extrinsics from the telemetry data (see Section 4.3). The existing NeRF prototype already does this — we can reuse that implementation directly.

---

## 3. Training Output Format

### 3.1 Target Directory Structure

```
datasets/agricultural_lora/
├── videos/
│   ├── gadget113_20250507_crop1_001.mp4      # 1280x720, 10fps, 3-5s, H.264
│   ├── gadget113_20250507_crop1_002.mp4
│   ├── gadget113_20250507_navfront_001.mp4
│   └── ...
├── metas/
│   ├── gadget113_20250507_crop1_001.txt       # Text caption
│   ├── gadget113_20250507_crop1_002.txt
│   └── ...
├── t5_xxl/                                    # Auto-generated by pre-compute script
│   ├── gadget113_20250507_crop1_001.pickle
│   └── ...
└── control_depth/
    ├── gadget113_20250507_crop1_001.mp4       # Normalized depth, same resolution/fps
    └── ...
```

### 3.2 File Naming Convention

```
{robot}_{date}_{camera}_{sequence}.mp4

Examples:
  gadget113_20250507_crop1_001.mp4
  gadget113_20250507_crop2_003.mp4
  gadget113_20250507_navfront_001.mp4
  demeter117_20250418_crop1_002.mp4
```

This naming encodes provenance (which robot, when, which camera) while staying filesystem-friendly. The sequence number increments for clips extracted from the same source capture.

### 3.3 Video Specifications

| Property | Requirement | Notes |
|----------|-------------|-------|
| **Resolution** | 1280x720 | Crop captures are native; nav cameras need upscale+pad |
| **Frame Rate** | 10 FPS | Downsample from 15fps streaming or interpolate from 5fps capture |
| **Duration** | 10 seconds | 100 frames (93 minimum required by Cosmos training) |
| **Codec** | H.264 (libx264) | MP4 container |
| **Pixel Format** | yuv420p | Standard for Cosmos |
| **CRF** | 18 | High quality |

### 3.4 Caption Specifications

Each clip gets a `.txt` file with a natural language description, 30-100 words. Captions are assembled programmatically from a metadata record per clip, then optionally refined.

See Section 4.4 for the captioning pipeline.

---

## 4. Data Processing Pipeline

### 4.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PHASE 1: SETUP (one-time)                       │
│                                                                     │
│  1a. Build robot-to-farm/crop manifest (human-provided YAML)        │
│  1b. Enumerate all captures in S3 → capture_index.csv               │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 PHASE 2: EXTRACT + CONVERT (per capture)            │
│                                                                     │
│  2a. Download + extract tar from S3                                 │
│  2b. For each camera in capture:                                    │
│      - Segment color video into 3-5s clips at 10fps, 1280x720      │
│      - Segment depth video into matching clips, normalize for Cosmos│
│      - Extract camera angle from telemetry extrinsics (see 4.3)    │
│  2c. Derive lighting condition from capture timestamp               │
│  2d. Write per-clip metadata JSON (intermediate format)             │
│  2e. Log warnings for any missing/unresolvable metadata (see 4.7)  │
│  2f. Clean up extracted tar                                         │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│              PHASE 3: CAPTION GENERATION (batch)                    │
│                                                                     │
│  3a. Assemble caption from metadata fields using template           │
│  3b. Optional: VLM refinement pass on a sample frame per clip       │
│  3c. Write .txt caption files to metas/                             │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│              PHASE 4: VALIDATE + FINALIZE                           │
│                                                                     │
│  4a. Validate all videos (resolution, fps, frame count, codec)      │
│  4b. Validate 1:1 video:caption:depth mapping                       │
│  4c. Spot-check depth alignment on random sample                    │
│  4d. Review preprocessing_report.json for warnings/skips            │
│  4e. Generate dataset statistics (diversity report)                  │
│  4f. Pre-compute T5 embeddings                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Phase 1: Setup — Robot/Farm Manifest

A human must provide a YAML manifest that maps robots to their deployment context. This is the only step that cannot be automated. Much of the S3 folder structure encodes this implicitly (e.g., `bowles-data/` suggests a farm called Bowles), and robot names map to specific deployments.

```yaml
# robot_manifest.yml
#
# NOTE: A robot may have been deployed on different fields/crops at different
# times. Where possible, use date ranges or S3 path context (e.g., bowles-data/)
# to disambiguate. For robots with a single known deployment, a simple entry
# suffices. For robots that moved between fields, use the deployments list.
# When building the initial manifest, be selective — start with robots whose
# crop assignment is confidently known for their capture date range.

robots:
  gadget-113:
    farm: "bowles"
    location: "central valley, CA"
    crop: "cotton"
    row_spacing_cm: 76
    notes: "Standard weeding config, 2x payload"
  demeter-117:
    farm: "bowles"
    location: "central valley, CA"
    crop: "cotton"
    row_spacing_cm: 76
  gopherus-111:
    # This robot was on different fields at different times
    deployments:
      - { start: "2025-02-01", end: "2025-06-30", farm: "delta", crop: "soybean" }
      - { start: "2025-07-01", end: "2025-11-30", farm: "bowles", crop: "cotton" }
    row_spacing_cm: 76
```

**Deriving crop type from S3 structure:** The `bowles-data/` prefix in S3 groups captures by farm, with robot numbers as sub-keys (e.g., `bowles-data/103/`, `bowles-data/111/`). Cross-referencing a robot's presence in a farm-specific prefix with its capture dates can help confirm or disambiguate crop assignments.

**What this provides for captions:** Crop type, farm/region. These fields would otherwise be impossible to get automatically.

### 4.3 Phase 2: Camera Angle Determination

For nav cameras, orientation is known from the name: `nav_front_0` is forward-facing, `nav_rear_0` is rear-facing.

For crop/payload cameras, we extract orientation from the telemetry extrinsics. The NeRF prototype (`ml-aigen-tools/prototypes/NeRF/`) already implements this exact flow and serves as the implementation reference:

1. Load the `data/chunk_*` files using `ChunkProcessor`
2. Process `/frames/*` messages through `TransformManager` to build the transform graph
3. Query camera pose as a quaternion `(qw, qx, qy, qz)` at any frame timestamp
4. Convert quaternion to pitch angle:
   - Pitch near -90 degrees = **top-down**
   - Pitch near -45 degrees = **angled/intra-row**
   - Pitch near 0 degrees = **forward-facing**

A single frame's transform is sufficient — no need to process the entire sequence. The relevant code path is `create_nerf_dataset.py` which already calls `TransformManager` to get camera poses for each frame and converts them to 4x4 matrices via `pose_to_matrix()`. We extract the pitch angle from the rotation component of that matrix.

This requires the `pyzome` library (already a dependency in the NeRF prototype).

**Fallback:** If telemetry chunks are missing or corrupted for a specific capture, log a warning and fall back to VLM inference on a single extracted frame (see Section 4.8 on logging).

### 4.4 Phase 3: Caption Generation

#### Metadata Assembly

For each clip, we assemble a metadata record from multiple sources:

| Field | Source | Automation Level |
|-------|--------|-----------------|
| `crop_type` | `robot_manifest.yml` via robot name + capture date | Fully automatic once manifest exists |
| `camera_name` | Filename parse (`nav_front_0`, `crop_center_1`, etc.) | Fully automatic |
| `camera_angle` | Telemetry extrinsics (crop) or camera name (nav) | Automatic |
| `view_type` | Derived from camera_name + angle | Automatic |
| `lighting` | Capture timestamp → time of day | Automatic |
| `robot_name` | S3 path / folder name | Fully automatic |
| `farm` | `robot_manifest.yml` | Fully automatic once manifest exists |
| `capture_date` | Folder timestamp | Fully automatic |

**Lighting derivation from timestamp:**

```python
# The folder name encodes UTC time: 2025-11-24-T16:47:59UTC
# Convert to local time for the farm location, then bucket:
LIGHTING_BUCKETS = {
    (5, 7):    "dawn, low warm light with long shadows",
    (7, 9):    "morning light with soft directional shadows",
    (9, 11):   "late morning light",
    (11, 14):  "midday with harsh overhead light and minimal shadows",
    (14, 16):  "afternoon light",
    (16, 18):  "golden hour with warm low-angle light",
    (18, 20):  "dusk, fading light",
}
# Overcast detection requires VLM or is omitted (acceptable for LoRA)
```

#### Caption Template

```python
CAPTION_TEMPLATE = """\
A {camera_angle} camera view from an agricultural robot \
{motion_description}in a {crop_type} field. \
{lighting_description}. \
The camera is a {camera_description} on the robot, \
viewing {view_description}.\
"""

# camera_angle: "forward-facing" | "rear-facing" | "top-down" | "angled downward"
# motion_description: "moving forward through crop rows " (nav) | "" (crop/stationary relative)
# crop_type: from manifest (date-aware for robots with multiple deployments)
# lighting_description: from timestamp bucket
# camera_description: "navigation camera" | "payload-mounted crop camera"
# view_description: based on camera type
```

#### Example Generated Captions

**Crop camera, top-down:**
```
A top-down camera view from an agricultural robot in a cotton field.
Late morning light. The camera is a payload-mounted crop camera on
the robot, viewing the canopy and soil between rows from directly above.
```

**Nav camera, forward:**
```
A forward-facing camera view from an agricultural robot moving forward
through crop rows in a soybean field. Morning light with soft
directional shadows. The camera is a navigation camera on the robot,
viewing the row corridor ahead.
```

These are 25-40 words — concise but informative enough for Cosmos T5 conditioning. They deliberately omit fields we can't reliably determine (growth stage, weather, weed pressure, soil type), avoiding hallucinated detail. A future VLM enrichment pass could add growth stage, field conditions, and other visual details.

#### Optional: VLM Enrichment Pass

For higher-quality captions, run a VLM on one frame from each clip to add visual details:

```
Given this frame from a {camera_angle} camera on an agricultural robot
in a {crop_type} field, describe in 2-3 sentences what you observe about:
the plant density, weed presence, soil visibility, and any notable
field conditions. Be factual and specific.
```

Append the VLM response to the template-generated caption. This adds cost ($0.01-0.03 per clip at typical VLM pricing) but can improve caption quality. For a 5,000-clip dataset, budget ~$50-150 for VLM enrichment.

### 4.5 Video Conversion

#### Crop Camera Captures (native 1280x720)

The crop cameras capture at exactly 1280x720 — no spatial transformation needed, just frame rate conversion and re-encoding:

```bash
# Segment into 10-second clips at 10fps from 5fps capture source
# -ss: start time, -t: duration
ffmpeg -i capture_color_crop_center_1_*.mp4 \
  -ss 0 -t 10 \
  -r 10 \
  -c:v libx264 -preset medium -crf 18 \
  -pix_fmt yuv420p \
  -an \
  output_clip_001.mp4
```

Note: The source capture is 5fps. Upsampling to 10fps will duplicate frames (each source frame appears twice). This is acceptable for Cosmos training — the model sees consistent visual content. Alternatively, use the `video/` folder recordings which are 15fps, but at lower 640x480 resolution.

**Recommendation:** Use `capture/` videos (5fps, 1280x720) for crop cameras. Frame duplication at 10fps is preferable to upscaling 640x480 video.

**Minimum frame requirement:** Cosmos training requires at least 93 frames per clip. At 10fps, this means clips must be at least 9.3 seconds. We target 10-second clips (100 frames) to have margin.

#### Nav Camera Captures (640x480 → 1280x720)

Nav cameras require upscaling and letterboxing:

```bash
# Scale 640x480 (4:3) → 960x720 (maintain aspect) then pad to 1280x720
ffmpeg -i capture_color_nav_front_0_*.mp4 \
  -ss 0 -t 10 \
  -vf "scale=960:720,pad=1280:720:160:0:black" \
  -r 10 \
  -c:v libx264 -preset medium -crf 18 \
  -pix_fmt yuv420p \
  -an \
  output_clip_001.mp4
```

This produces 1280x720 with 160px black bars on left and right. An alternative is to crop to 16:9 from the center of the 4:3 frame, losing top/bottom content:

```bash
# Crop 640x480 → 640x360 (16:9 center crop), then scale to 1280x720
ffmpeg -i capture_color_nav_front_0_*.mp4 \
  -ss 0 -t 10 \
  -vf "crop=640:360:0:60,scale=1280:720" \
  -r 10 \
  -c:v libx264 -preset medium -crf 18 \
  -pix_fmt yuv420p \
  -an \
  output_clip_001.mp4
```

**Recommendation:** Use center-crop + upscale for nav cameras. Padding introduces artificial black regions that would confuse the model. Cropping loses some vertical FOV but maintains natural image content across the full frame. Apply the same crop to the corresponding depth video.

#### Depth Video Processing

Depth captures are already 8-bit encoded H.264 at the same resolution as color. They need the same spatial transform, rate conversion, and Cosmos-specific normalization (invert so closer = brighter):

```bash
# For crop cameras (already 1280x720):
ffmpeg -i capture_depth_crop_center_1_*.mp4 \
  -ss 0 -t 10 \
  -vf "negate" \
  -r 10 \
  -c:v libx264 -preset medium -crf 18 \
  -pix_fmt yuv420p \
  -an \
  depth_clip_001.mp4
```

The `negate` filter inverts the grayscale so closer objects (higher depth value in source) become brighter, matching Cosmos convention where closer = brighter.

For nav cameras, apply the same crop+scale as the color video, plus negate.

### 4.6 Clip Segmentation Strategy

Each capture is ~13-15 seconds long. At 10 seconds per clip (required for 93+ frames at 10fps), most captures yield only 1 clip per camera. With 4 cameras per capture and ~5,000 captures:

| | Per Capture | Total (5K captures) |
|-|-------------|---------------------|
| Clips per camera | 1 (captures ≥10s) | 5,000-10,000 per camera |
| Cameras per capture | 4 | — |
| Total clips | ~4 | ~20,000-40,000 |

This is still well above the 5,000-clip target. We can be selective:
- **Start with crop cameras only** (best resolution, most interesting views) → ~10,000-20,000 candidate clips
- Sample a diverse subset of 5,000 clips across robots, dates, and cameras
- Later add nav camera clips to expand diversity

**Segmentation rules:**
- Single 10-second clip from the start of each capture (simpler than multiple segments)
- Skip captures shorter than 10 seconds entirely
- Skip clips where >20% of depth frames are black/zero (sensor dropout)
- Skip clips with significant motion blur (optional, detect via Laplacian variance)

**Re-processing from cached captures:** If captures have already been downloaded to a local cache (e.g., `/tmp/captures/`), the pipeline should detect and reuse them rather than re-downloading from S3. Only the clip extraction step needs to be re-run with the updated 10-second duration.

### 4.7 Logging and Warnings

The pipeline should log warnings whenever important metadata cannot be resolved, rather than silently producing incomplete or incorrect output. Use Python's `logging` module at WARNING level for recoverable issues and ERROR for skipped captures.

**Required warnings:**

| Condition | Severity | Action |
|-----------|----------|--------|
| Robot not found in manifest | WARNING | Caption will omit crop type; log robot name |
| Robot has multiple deployments and capture date falls outside all ranges | WARNING | Caption will omit crop type; log robot + date |
| Telemetry chunks missing or empty (`data/chunk_*` not found) | WARNING | Cannot determine crop camera angle; fall back to VLM or mark as "unknown" |
| Transform graph fails to resolve camera pose | WARNING | Log camera name + capture; fall back to VLM |
| Capture color video missing or corrupt | ERROR | Skip this camera for this capture |
| Capture depth video missing or corrupt | WARNING | Produce RGB clip without paired depth |
| Capture shorter than minimum clip duration (< 10s) | WARNING | Skip capture entirely |
| Metadata CSV missing (no frame timestamps) | WARNING | Cannot verify depth-color alignment; proceed with caution |
| S3 object in Glacier storage class | WARNING | Skip; log S3 key for later restore |

All warnings should be aggregated into a preprocessing report (`preprocessing_report.json`) that summarizes:
- Total captures processed vs. skipped
- Count of each warning type
- List of captures skipped with reasons
- List of robots missing from manifest

This report allows a human to review pipeline health and decide whether to update the manifest, restore Glacier objects, or investigate other issues before training.

### 4.8 Phase 4: Validation and Finalization

```python
# Validation checks for each clip:
def validate_clip(video_path, depth_path, caption_path):
    errors = []

    # Video checks
    cap = cv2.VideoCapture(str(video_path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if (w, h) != (1280, 720):
        errors.append(f"resolution {w}x{h}, expected 1280x720")
    if abs(fps - 10) > 0.5:
        errors.append(f"fps {fps}, expected 10")
    if frames < 93:
        errors.append(f"only {frames} frames, need >= 93")

    # Depth checks
    if not Path(depth_path).exists():
        errors.append("missing depth video")
    else:
        dcap = cv2.VideoCapture(str(depth_path))
        dframes = int(dcap.get(cv2.CAP_PROP_FRAME_COUNT))
        dcap.release()
        if dframes != frames:
            errors.append(f"depth has {dframes} frames, color has {frames}")

    # Caption check
    if not Path(caption_path).exists():
        errors.append("missing caption file")
    else:
        caption = Path(caption_path).read_text().strip()
        if len(caption.split()) < 15:
            errors.append(f"caption too short ({len(caption.split())} words)")

    return errors
```

**Dataset statistics to generate before training:**
- Distribution of clips per robot
- Distribution of clips per crop type
- Distribution across calendar months (seasonal diversity)
- Distribution across camera types (nav vs crop)
- Caption word count histogram

---

## 5. Training on DGX Spark

### 5.1 Pre-compute T5 Embeddings

```bash
python -m scripts.get_t5_embeddings \
  --dataset_path datasets/agricultural_lora/
```

### 5.2 Launch LoRA Training

```bash
# DGX Spark (single GPU, shared memory architecture)
CUDA_VISIBLE_DEVICES=0 python -m scripts.train \
  --config=cosmos_transfer2/configs/base/config.py \
  -- experiment=transfer2_agricultural_lora \
  model.config.train_architecture=lora \
  model.config.lora_rank=8 \
  dataloader_train.batch_size=1 \
  trainer.max_iter=2000
```

Note: With `batch_size=1` and `lora_rank=8`, this should fit within the DGX Spark's 128GB shared VRAM. If OOM occurs, reduce `lora_rank` to 4.

### 5.3 Checkpoint Conversion and Inference

```bash
# Convert DCP checkpoint to PyTorch format
CHECKPOINT_DIR=${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer/agricultural_lora/checkpoints/iter_002000

python scripts/convert_distcp_to_pt.py \
  $CHECKPOINT_DIR/model \
  $CHECKPOINT_DIR

# Inference
torchrun --nproc_per_node=1 examples/inference.py \
  your_spec.json \
  outputs/agricultural_test \
  --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
  --experiment transfer2_agricultural_lora
```

---

## 6. Data Diversity and Sampling

### 6.1 Priority Dimensions

Given the data we have, focus on maximizing diversity along these axes:

| Dimension | Available Variation | How to Ensure |
|-----------|-------------------|---------------|
| **Crop type** | Cotton, soybean, others per manifest | Sample proportionally across robots/farms |
| **Season** | Captures span Feb 2025 – Jan 2026 | Sample across calendar months (implicitly captures growth stage variation) |
| **Camera view** | Nav forward, nav rear, crop top-down, crop angled | Include all camera types; angle from telemetry |
| **Time of day** | Captures span full daylight hours (UTC timestamps) | Sample across time-of-day buckets |
| **Robot variety** | ~30 distinct robots | Sample across multiple robots (different sensor wear, calibration) |

### 6.2 Suggested Sampling for 5,000 Clips

| Camera | Clips | Rationale |
|--------|-------|-----------|
| `crop_center_1` (top-down) | 1,500 | Best resolution, highest value view |
| `crop_center_2` (top-down) | 1,500 | Second crop camera, same quality |
| `nav_front_0` (forward) | 1,000 | Different perspective, row corridors |
| `nav_rear_0` (rear) | 500 | Useful variety but similar to nav_front |
| `crop_*` (angled/intra-row) | 500 | If angle determination finds angled captures |

### 6.3 Data Splits

| Split | Clips | Selection |
|-------|-------|-----------|
| **Train** | 4,500 (90%) | Random sample respecting diversity |
| **Val** | 250 (5%) | Held-out robots or dates |
| **Test** | 250 (5%) | Completely unseen robots or dates |

Prefer splitting by **robot** or **date** rather than random, so validation truly tests generalization to new scenes rather than memorization of nearby frames.

---

## 7. Storage and Cost Estimates

### 7.1 Storage

| Data | Per Clip (3s) | 5,000 Clips |
|------|---------------|-------------|
| 720p RGB video (CRF 18, 10s) | ~12 MB | ~60 GB |
| Depth video (10s) | ~5 MB | ~25 GB |
| T5 embedding | ~1 MB | ~5 GB |
| Caption text | <1 KB | ~5 MB |
| **Total** | ~18 MB | **~90 GB** |

### 7.2 Processing Cost

| Step | Compute | Notes |
|------|---------|-------|
| S3 download + extract | Network-bound | ~200-400 GB of tars for 5K captures |
| FFmpeg conversion | CPU-bound | ~1-2s per clip; ~3 hrs for 5K clips (parallelizable) |
| Telemetry extrinsics extraction | CPU-bound | ~1-2s per capture (pyzome + chunk parsing) |
| VLM caption enrichment | API cost | ~$50-150 for 5K clips (optional, future) |
| T5 embedding pre-compute | GPU | ~30 min on DGX Spark |

---

## 8. Implementation Plan

### Step 1: Create robot manifest
- Survey S3 bucket for robot names and date ranges
- Cross-reference with farm-specific prefixes (e.g., `bowles-data/`) to confirm crop assignments
- Have a human fill in farm, crop type for each robot (with date ranges where deployments changed)
- Be selective: start with robots whose crop assignment is confidently known
- Validate by spot-checking a few captures per robot

### Step 2: Build capture index
- Enumerate all `.tar` files across `data-capture/{robot}/`
- Check storage class; log Glacier objects as inaccessible and skip them
- Parse timestamps, compute date ranges per robot
- Output: `capture_index.csv` with columns: `robot, s3_path, timestamp, date, storage_class`

### Step 3: Implement extraction + conversion pipeline
- Download tar, extract to temp dir
- For each camera: segment + convert color and depth videos
- Extract crop camera angle from telemetry extrinsics (reusing NeRF prototype code)
- Write intermediate metadata JSON per clip
- Log warnings for missing/corrupt data (see Section 4.7)
- Delete extracted tar

### Step 4: Generate captions
- Load robot manifest + clip metadata
- Resolve crop type using manifest (date-aware for multi-deployment robots)
- Apply caption template
- Log warnings for clips where crop type could not be resolved
- Optional: VLM enrichment pass on a sample for richer descriptions
- Write `.txt` files

### Step 5: Validate and finalize
- Run validation script on all clips
- Review `preprocessing_report.json` for warnings and skipped captures
- Generate diversity statistics
- Pre-compute T5 embeddings

### Step 6: Train
- Launch LoRA training on DGX Spark
- Monitor loss curves
- Evaluate on held-out test clips

---

## Appendix A: Existing Code References

The NeRF prototype at `~/Desktop/code/claude_code/ml-aigen-tools/prototypes/NeRF/` contains reusable components:

| File | Relevant Code | Use For |
|------|--------------|---------|
| `chunk_processor.py` | `ChunkProcessor` class: loads XYLEM chunks, parses topics, extracts video frames | Loading telemetry + video from captures |
| `transforms.py` | `TransformManager`: builds transform graph from `/frames/*` messages | Camera pose extraction (Approach A for angle) |
| `create_nerf_dataset.py` | `get_camera_intrinsics()`, `CAMERA_MAPPING`, `preprocess_depth_frame()` | Camera intrinsics, depth decoding constants |
| `examples/telemetry_data.md` | Full topic list, frame hierarchy docs | Understanding telemetry message structure |
| `examples/depth_encoding.md` | Depth quantization formula and per-camera max values | Correct depth decoding |

Key constants from the NeRF code:
```python
CAMERA_MAPPING = {
    'crop_center_1': {'frame': 'camera/crop/center@1', 'intrinsics_topic': '/camera/crop/center/intrinsics/color@1', 'max_depth_mm': 1020},
    'crop_center_2': {'frame': 'camera/crop/center@2', 'intrinsics_topic': '/camera/crop/center/intrinsics/color@2', 'max_depth_mm': 1020},
    'nav_front_0':   {'frame': 'camera/nav/front',     'intrinsics_topic': '/camera/nav/front/intrinsics/color',     'max_depth_mm': 2550},
    'nav_rear_0':    {'frame': 'camera/nav/rear',       'intrinsics_topic': '/camera/nav/rear/intrinsics/color',      'max_depth_mm': 2550},
}
```

## Appendix B: Multiview Workflow (Future)

The full ControlNet multiview workflow (training custom camera geometry for cross-view synthesis) is deferred to a later phase. It requires:
- 8x H100 GPUs (AWS p5.48xlarge)
- Synchronized multi-camera clips (all 4 cameras, temporally aligned)
- 480p resolution (854x480)
- 10,000-50,000+ synchronized clip sets
- Per-camera intrinsics and extrinsics from telemetry

The data pipeline built here (S3 extraction, metadata assembly, caption generation) will be reusable. The main additions for multiview are: temporal synchronization of cross-camera clips using the metadata CSV timestamps, and inclusion of camera pose matrices in the training data.

## Appendix C: Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Resolution mismatch | Video not exactly 1280x720 | Use ffmpeg with pad/crop filter |
| Frame count mismatch | Variable frame rate source | Re-encode with `-r 10` |
| Data loader crash | Missing caption file | Validate 1:1 video:caption mapping |
| OOM on DGX Spark | Batch size too large | Set `batch_size=1`, reduce `lora_rank` to 4 |
| Poor generation quality | Captions too generic | Add VLM enrichment pass |
| Depth misaligned | Color/depth from different time windows | Ensure same `-ss`/`-t` for both |
| Black bars in nav clips | Padding instead of cropping | Use center-crop + scale approach |
| Source video too short | Capture < 10 seconds | Skip; use only captures ≥ 10s |
| Corrupt tar in S3 | Interrupted upload | Log and skip; plenty of data |
