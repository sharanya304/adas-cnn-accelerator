import os
import time
import cv2
import torch
import torch.nn as nn
import numpy as np

# Import the model extractor and config directly from your file
from extract_patch_canvas import (
    MCUNetV2_CanvasExtractor,
    CHECKPOINT_PATH,
    KITTI_IMG_PATH,
    TILE_DIM,
    OVERLAP,
    DEVICE
)

def profile_mcunetv2_performance():
    print("=" * 60)
    print("       MCUNETV2 REAL IMAGE PERFORMANCE PROFILING")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # Ensure temporary dump directory exists so np.save won't crash
    temp_dir = "./temp_profile_dump"
    os.makedirs(temp_dir, exist_ok=True)

    # 1. Initialize Model
    extractor = MCUNetV2_CanvasExtractor(tile_dim=TILE_DIM, overlap=OVERLAP).to(DEVICE)
    
    # Check absolute vs relative checkpoint path safety
    ckpt_path = CHECKPOINT_PATH
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.basename(CHECKPOINT_PATH)

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
        patch_weights = {k.replace('patch_stage.', '').replace('module.', ''): v 
                         for k, v in state_dict.items() if 'patch_stage.' in k}
        extractor.patch_stage.load_state_dict(patch_weights, strict=True)
        print(f"[INFO] Loaded checkpoint successfully from: {ckpt_path}")
    else:
        print(f"[WARNING] Checkpoint not found at {ckpt_path}. Running with initialized weights.")

    extractor.eval()

    # 2. Load and Preprocess target image (KITTI 000129.png)
    if not os.path.exists(KITTI_IMG_PATH):
        raise FileNotFoundError(f"[ERROR] Could not find target image at '{KITTI_IMG_PATH}'")

    img_bgr = cv2.imread(KITTI_IMG_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 192))

    real_image_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    real_image_tensor = real_image_tensor.to(DEVICE)

    # 3. Hook Intermediate Feature Maps
    captured_tensors = {}
    hooks = []

    def make_hook(layer_name):
        def hook(module, input, output):
            captured_tensors[layer_name] = output.shape
        return hook

    for name, module in extractor.named_modules():
        if isinstance(module, nn.Conv2d) or name == "patch_stage":
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Warmup Run
    with torch.no_grad():
        _ = extractor.extract_canvas(real_image_tensor, out_dir=temp_dir)

    # 4. Profile Latency & Peak Memory Allocation
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        global_canvas_tensor = extractor.extract_canvas(real_image_tensor, out_dir=temp_dir)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_memory_mb = sum(p.numel() * p.element_size() for p in extractor.parameters()) / (1024 ** 2)

    total_latency_ms = (time.perf_counter() - start_time) * 1000.0
    throughput_fps = 1000.0 / total_latency_ms if total_latency_ms > 0 else 0.0

    # Clean up hooks & temporary folders
    for h in hooks:
        h.remove()

    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)

    # 5. Output Formatted Results
    print("\n" + "=" * 60)
    print("      QUANTITATIVE PERFORMANCE METRICS (COPY TO SLIDE)")
    print("=" * 60)
    print(f"  Execution Time / Latency  : {total_latency_ms:.2f} ms")
    print(f"  Throughput                : {throughput_fps:.2f} samples/s")
    print(f"  Peak Memory Requirement   : {peak_memory_mb:.2f} MB")
    print(f"  Stitched Canvas Shape     : {list(global_canvas_tensor.shape)}")
    print(f"  Total Layers Profiled     : {len(captured_tensors)} operations")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    profile_mcunetv2_performance()