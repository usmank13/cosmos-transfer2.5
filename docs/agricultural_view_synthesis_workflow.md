# Using Cosmos Transfer 2.5 for Agricultural View Synthesis

## Problem Statement

**Goal:** Transform top-down camera video from an autonomous agricultural robot in soybean fields into an angled (~45° roll) view that sees better under plant canopies.

**Available Data:**
- Top-down soybean field videos with Intel RealSense depth
- Abundant angled-view video data of other crops (e.g., cotton)
- Multiple synchronized cameras on the robot (laterally spaced top-down + forward-facing nav camera)
- Trained NeRF providing 3D scene representation

---

## Is Cosmos Transfer 2.5 a Good Fit?

### What Cosmos Transfer Does

Cosmos Transfer 2.5 is a **multi-controlnet video generation model** designed for physical AI applications. It excels at:

- **Sim2Real augmentation**: Transforming synthetic/structured data into realistic video
- **Structured conditioning**: Generating video that matches control signals (depth, edges, segmentation)
- **Multi-view consistency**: Generating synchronized video across multiple camera viewpoints
- **Temporal consistency**: Maintaining coherence across video frames (unlike per-frame diffusion)

### What Cosmos Transfer Does NOT Do

- **Novel view synthesis from 2D**: It cannot infer 3D geometry from a single 2D view and render from a new angle
- **Geometric transformation**: Control inputs describe the OUTPUT view's structure, not a source to transform

### The Key Insight

Cosmos requires you to provide control signals (depth, edges, etc.) that describe the **target view's structure**. If you can generate that structure from another source (like a NeRF), Cosmos can render it photorealistically.

---

## Viable Workflow

### Core Idea

```
NeRF (provides geometry)  +  Cosmos (provides photorealism)
         ↓                            ↓
   Depth from angled view    →    Realistic crop video
```

### Data Flow

```
1. Train NeRF on RealSense RGB+depth data (top-down views)
2. Render NeRF from target angled camera pose → depth maps
3. Pass depth as control input to Cosmos
4. Use soybean reference image for appearance guidance
5. Cosmos generates photorealistic angled-view video
```

---

## Key Cosmos Mechanisms

### Temporal Consistency

Unlike per-frame diffusion models, Cosmos processes video holistically:

- All frames encoded into shared latent space via VAE
- Diffusion operates on **full video latent**, not individual frames
- For longer videos: autoregressive mode with overlapping chunks
- Previous frames condition next chunk, maintaining continuity

### Cross-View Consistency (Multiview)

When generating multiple camera views simultaneously:

- **CrossViewAttention** layers let views share spatial information
- Configurable attention map defines which cameras "see" each other
- Single shared text caption anchors all views to same scene
- Per-view depth ensures geometric consistency

### Control Conditioning

| Control Type | Purpose | Source |
|--------------|---------|--------|
| **Depth** | 3D spatial structure | NeRF render, RealSense |
| **Edge** | Structural boundaries | Computed on-the-fly |
| **Segmentation** | Semantic regions | SAM2 or pre-computed |
| **Image Context** | Appearance/style | Reference image (SigLip2 encoded) |

---

## Proposed Pipeline

### Phase 1: Quick Test (No Finetuning)

Test the base model to see how well it handles agricultural scenes.

#### Step 1: Prepare Depth from NeRF

```python
import numpy as np

# Render NeRF from target angled camera pose
depth_frames = []
for frame_idx in range(num_frames):
    nerf_output = model.render(angled_camera_pose[frame_idx])
    depth_frames.append(nerf_output['depth'])  # (H, W)

depth_video = np.stack(depth_frames)  # (T, H, W), float32
np.savez_compressed('angled_view_depth.npz', depth=depth_video)
```

#### Step 2: Select Reference Image

Options for `image_context_path`:
- **Your top-down soybean frame** (recommended): Correct crop appearance, your sensor's color profile
- **Online angled soybean image**: Correct viewing angle but different field/lighting
- **Combination approach**: Test both and compare results

#### Step 3: Create Inference Config

```json
{
    "name": "soybean_angled_test",
    "depth": {
        "control_path": "/path/to/angled_view_depth.npz",
        "control_weight": 1.0
    },
    "image_context_path": "/path/to/topdown_soybean_frame.jpg",
    "prompt": "agricultural field with soybean plants, angled camera perspective showing crop rows and under-canopy view, realistic outdoor lighting, green vegetation",
    "negative_prompt": "blurry, distorted, unrealistic, cartoon",
    "num_steps": 35,
    "guidance": 7.5
}
```

#### Step 4: Run Inference

```bash
python examples/inference.py \
    -i soybean_test_config.json \
    -o output/soybean_angled_test/
```

### Phase 2: Finetuning (If Needed)

If base model results are poor on agricultural scenes, finetune on your angled-view crop data.

#### Training Data Structure

```
dataset/
├── videos/
│   └── *.mp4                    # Angled-view crop videos
├── control_input_depth/
│   └── *.npz                    # RealSense depth (matching videos)
└── captions/
    └── *.json                   # Text descriptions
```

#### Caption Format

```json
{
    "caption": "angled camera view of cotton field, green plants in rows, outdoor agricultural setting"
}
```

#### Run Finetuning

```bash
torchrun --nproc_per_node=8 --master_port=12341 \
    -m scripts.train \
    --config=cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py \
    experiment=agricultural_angled_view \
    job.wandb_mode=disabled
```

### Phase 3: Multi-View Scene Generation

Generate multiple consistent angled views simultaneously.

#### Custom Camera Configuration

Modify `cosmos_transfer2/multiview_config.py`:

```python
MULTIVIEW_CAMERA_KEYS: tuple[str, ...] = (
    "angled_left",
    "angled_center",
    "angled_right",
)
```

#### Cross-View Attention Map

Define which cameras share information:

```python
cross_view_attn_map = {
    "angled_center": ["angled_left", "angled_right"],
    "angled_left": ["angled_center"],
    "angled_right": ["angled_center"],
}
```

#### Multi-View Inference Config

```json
{
    "name": "soybean_multiview",
    "prompt": "soybean agricultural field, angled camera view",
    "angled_left": {
        "control_path": "nerf_depth_angled_left.npz"
    },
    "angled_center": {
        "control_path": "nerf_depth_angled_center.npz"
    },
    "angled_right": {
        "control_path": "nerf_depth_angled_right.npz"
    },
    "num_conditional_frames": 1,
    "num_steps": 35
}
```

---

## Depth Format Requirements

### NPZ Format (Recommended)

```python
# Shape: (T, H, W) for video, (H, W) for single frame
# Type: float32
# Values: Any scale (normalized internally)
# NaN values treated as max depth

depth_array = np.array(depth_frames, dtype=np.float32)
np.savez_compressed('depth.npz', depth=depth_array)
```

### Resolution

- Should match target video resolution (will be resized if not)
- Typical: 720p (1280x720) or 480p (854x480)

### Temporal Alignment

- Frame N of depth must correspond to frame N of output video
- For NeRF renders: use consistent camera trajectory

---

## Hardware Requirements

| Task | GPUs | VRAM per GPU |
|------|------|--------------|
| Single-view inference | 1 | ~65 GB (H100/A100 80GB) |
| Multi-view inference (7 views) | 8 | ~65 GB each |
| Finetuning | 8 | ~80 GB each |

---

## Expected Outcomes

### What Should Work Well

- **Geometry preservation**: Depth control maintains plant positions, heights, canopy structure
- **Appearance transfer**: Image context + prompt guides crop-specific appearance
- **Temporal consistency**: Smooth video without per-frame artifacts
- **Cross-view consistency**: (with multiview) Same plants appear consistently across cameras

### Potential Challenges

1. **Domain gap**: Base model trained on driving/urban scenes, not agriculture
   - *Mitigation*: Finetune on your angled crop data

2. **NeRF quality**: Artifacts in NeRF depth will propagate
   - *Mitigation*: Use high-quality NeRF, smooth depth maps

3. **Under-canopy hallucination**: Regions occluded in top-down view have no ground truth
   - *Mitigation*: Cosmos will hallucinate based on depth cues and learned priors

4. **Crop generalization**: Trained on cotton, generating soybean
   - *Mitigation*: Strong image context + detailed prompts; crops are structurally similar

---

## Quick Reference: Key Files

| Purpose | File |
|---------|------|
| Single-view inference | `examples/inference.py` |
| Multi-view inference | `cosmos_transfer2/multiview.py` |
| Multi-view config | `cosmos_transfer2/multiview_config.py` |
| Training config | `cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/` |
| Control input processing | `cosmos_transfer2/_src/transfer2/datasets/augmentors/control_input.py` |
| Depth loading | `cosmos_transfer2/_src/imaginaire/datasets/webdataset/decoders/depth.py` |

---

## Summary

**The viable approach:**

1. Use your NeRF to render depth from the desired angled viewpoint
2. Provide that depth as control input to Cosmos
3. Use top-down soybean images as appearance reference
4. Generate photorealistic angled-view video

**Cosmos provides:**
- Photorealistic rendering from structured depth
- Temporal consistency across video frames
- Cross-view consistency for multi-camera setups
- Appearance conditioning via reference images

**You provide:**
- 3D geometry via NeRF-rendered depth
- Crop appearance via reference images
- Scene description via text prompts
