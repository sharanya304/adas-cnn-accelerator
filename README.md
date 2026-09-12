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
python train_mcunetv2.py
```
Runs with default settings: 40 epochs, 128×128 tile size, 16px overlap,
learning rate 0.0005, confidence threshold 0.5. Trains `MCUNetV2_Detector`
(patch_stage + layer_stage backbone, cls_head heatmap, wh_head box-size
regression) on a 70/30 train/test split (seed=42).

### 2. Verify tiling/stitching on a single image
```bash
python extract_patch_canvas.py
```
Loads the trained checkpoint's `patch_stage` weights, tiles one KITTI image,
runs each tile through the backbone, and stitches outputs into one global
feature canvas — confirming tiled computation matches whole-image computation.

### 3. Profile tiling/stitching performance
```bash
python profile_performance.py
```
Runs a warmup pass + a timed pass of the patch tiling/stitching stage and
prints latency, throughput, and peak memory.

### 4. Export quantized weights
```bash
python fixed_point.py
```
Converts the trained checkpoint's weights to Q8.8 fixed-point integers,
grouped by stage, and writes a human-readable `fixed_point_weights.txt`.

## Input Format

- **Dataset:** KITTI tracking dataset, expected layout:
