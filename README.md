# ADAS CNN Object Detection Accelerator — Phase 1 Reference Model

Phase 1 Python reference-model code for a CNN-based object detection accelerator
targeting a 16×16 weight-stationary systolic array for ADAS applications.
Implements an MCUNetV2-inspired patch-based detector with a CenterNet-style
heatmap + width/height regression head, using tiled/stitched feature
computation to explore hardware-relevant tiling ahead of RTL implementation.

## Software Required

- Python 3.9+
- PyTorch (CUDA-enabled recommended — tested on NVIDIA GeForce RTX 2050)
- numpy
- opencv-python (`cv2`)

Install dependencies:
```bash
pip install torch numpy opencv-python
```

## How to Execute

### 1. Train the detector
```bash
python train_mcunetv2.py --dataset_dir <path_to_KITTI_tracking_dir> --epochs 40
```
Optional CLI flags (all have defaults):

| Flag | Default | Purpose |
|---|---|---|
| `--dataset_dir` | `D:\CNN_python_training\training` | Root KITTI tracking folder (must contain `image_02/` and `label_02/`) |
| `--epochs` | 40 | Training epochs |
| `--tile_dim` | 128 | Patch tile size (px) |
| `--overlap` | 16 | Tile overlap (px) |
| `--lr` | 0.0005 | AdamW learning rate |
| `--thresh` | 0.5 | Detection confidence threshold used when generating preview images |
| `--output_dir` | `./train_mcunetv2_output` | Where weights + previews are saved |
| `--num_previews` | -1 (all) | Number of test-set preview images to generate |

Trains `MCUNetV2_Detector` (patch_stage + layer_stage backbone, cls_head heatmap,
wh_head box-size regression) on a 70/30 train/test split (seed=42).

### 2. Verify tiling/stitching on a single image
```bash
python extract_patch_canvas.py
```
Loads a trained checkpoint's `patch_stage` weights, tiles one KITTI image,
runs each tile through the backbone, and stitches outputs into one global
feature canvas — confirming tiled computation matches whole-image computation.

> **Edit before running:** `CHECKPOINT_PATH` and `KITTI_IMG_PATH` are hardcoded
> at the top of the script.

### 3. Profile tiling/stitching performance
```bash
python profile_performance.py
```
Runs one warmup pass + one timed pass of the patch tiling/stitching stage
(imports its model/config directly from `extract_patch_canvas.py`) and prints
latency, throughput, and peak memory.

### 4. Export quantized weights
```bash
python fixed_point.py
```
Converts a trained checkpoint's weights to Q8.8 fixed-point integers, grouped
by stage (`patch_stage`, `layer_stage`, `head_trunk`, `cls_head`, `wh_head`),
and writes a human-readable `fixed_point_weights.txt`.

> **Edit before running:** `CHECKPOINT_PATH` at the bottom of the script
> (`mcunetv2_40epochs.pth`) assumes the file is in the current directory.

## Input Format

- **Dataset:** KITTI tracking dataset, expected layout:
