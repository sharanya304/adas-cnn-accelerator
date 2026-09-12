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
- <dataset_dir>/image_02/<sequence>/<frame>.png
- <dataset_dir>/label_02/<sequence>.txt
- - **Label columns used:** frame index, class (col 3), bbox left/top/right/bottom (cols 7–10) — standard KITTI tracking format
- **Classes:** Car/Van/Truck → `Car`, Cyclist → `Cyclist`, Pedestrian/Person_sitting → `Pedestrian` (3-class subset of the full 8 KITTI classes)
- **Image preprocessing:** resized to 640×192, normalized to [0, 1], BGR→RGB

- ## Expected Output

- `train_mcunetv2.py` → trained weights at `<output_dir>/mcunetv2_<epochs>epochs.pth`; per-epoch focal + WH loss printed to console; bounding-box preview images at `<output_dir>/preview_detections/preview_N_<frame>_bbox.png`
- `extract_patch_canvas.py` → per-tile and full-canvas feature tensors (`.npy` + `.txt`) in `./canvas_dump_<seq>_<frame>/`
- `profile_performance.py` → printed latency (ms), throughput (samples/s), peak memory (MB), stitched canvas shape
- `fixed_point.py` → `fixed_point_weights.txt` with Q8.8 quantized weights per stage
- Sample outputs included in `sample_outputs/` in this repo

## Important Parameters

| Parameter | Value | Notes |
|---|---|---|
| Tile size / overlap | 128×128 / 16px | CLI-configurable; script warns if overlap is smaller than the estimated receptive field |
| Patch backbone downsample | 8× | `patch_stage`: stem conv + 3 inverted-residual blocks |
| Total network downsample | 32× | `patch_stage` (8×) × `layer_stage` (4×) — sets detection grid size |
| Input resolution | 640×192 | Fixed resize target |
| Epochs / LR | 40 / 5e-4 | AdamW, weight_decay=1e-4 |
| Confidence threshold | 0.5 (default) | CLI `--thresh` |
| Loss function | CenterNet focal loss (α=2, β=4) + WH L1 loss | WH loss weighted 0.1× |
| Heatmap Gaussian radius | Fixed, radius=2 | Same radius applied to every object regardless of class or box size |
| Min regressed box size | 4 px | Guards against degenerate near-zero predictions early in training |
| Train/test split | 70% / 30%, seed=42 | |
| Quantization | Q8.8 fixed-point | 16-bit signed, 8 fractional bits |
