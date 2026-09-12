import os
import cv2
import torch
import torch.nn as nn
import numpy as np

# =====================================================================
# 1. SETUP & CONFIGURATION
# =====================================================================
CHECKPOINT_PATH = "train_mcunetv2_output/mcunetv2_40epochs.pth"

# Exact path from your screenshot containing cars, pedestrians, and cyclists
KITTI_IMG_PATH = r"D:\CNN_python_training\training\image_02\0002\000129.png"
OUTPUT_DIR = "./canvas_dump_0002_000129"

TILE_DIM = 128
OVERLAP = 16

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Execution Device: {DEVICE}")

def dump_feature_tensor(tensor, name, out_dir):
    """
    Saves ONE feature-map tensor (patch contribution or the full stitched
    global canvas) in two forms:
      - <name>.npy  : exact float32 values for numpy/binary verification
      - <name>.txt  : human-readable printout (shape header + values)
    """
    arr = tensor.detach().cpu().numpy()[0]  # Drop batch dimension -> Shape: (C, H, W)
    np.save(os.path.join(out_dir, f"{name}.npy"), arr)
    with open(os.path.join(out_dir, f"{name}.txt"), "w") as f:
        f.write("=" * 72 + "\n")
        f.write(f" FEATURE TENSOR: {name}\n")
        f.write(f" Shape (C, H, W): {list(arr.shape)}\n")
        f.write("=" * 72 + "\n\n")
        f.write(np.array2string(arr, threshold=np.inf, max_line_width=120))
        f.write("\n")

# =====================================================================
# 2. MCUNetV2 ARCHITECTURE WITH HOOKED CANVAS EXTRACTION
# =====================================================================
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, kernel_size):
        super(InvertedResidualBlock, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        hidden_dim = int(in_channels * expand_ratio)
        padding = kernel_size // 2

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class MCUNetV2_CanvasExtractor(nn.Module):
    def __init__(self, tile_dim=128, overlap=16):
        super(MCUNetV2_CanvasExtractor, self).__init__()
        self.tile_dim = tile_dim
        self.overlap = overlap
        self.stride = tile_dim - overlap
        self.patch_ds_factor = 8  # 8x downsampling factor

        self.patch_stage = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            InvertedResidualBlock(16, 24, stride=2, expand_ratio=1, kernel_size=3),
            InvertedResidualBlock(24, 32, stride=1, expand_ratio=3, kernel_size=1),
            InvertedResidualBlock(32, 40, stride=2, expand_ratio=3, kernel_size=3),
        )

    def _tile_positions(self, total_size):
        starts = list(range(0, total_size - self.tile_dim + 1, self.stride))
        if not starts or starts[-1] != total_size - self.tile_dim:
            starts.append(total_size - self.tile_dim)

        ds = self.patch_ds_factor
        out_tile_size = self.tile_dim // ds
        entries = []
        filled_up_to = 0
        for s in starts:
            out_start = s // ds
            fill_start = max(out_start, filled_up_to)
            fill_end = out_start + out_tile_size
            if fill_end <= fill_start:
                continue
            local_start = fill_start - out_start
            local_end = fill_end - out_start
            entries.append((s, fill_start, fill_end, local_start, local_end))
            filled_up_to = fill_end
        return entries

    def extract_canvas(self, full_image, out_dir):
        B, C, H, W = full_image.shape
        ds = self.patch_ds_factor

        canvas_h, canvas_w = H // ds, W // ds
        stitched_canvas = torch.zeros((B, 40, canvas_h, canvas_w), device=full_image.device)

        y_entries = self._tile_positions(H)
        x_entries = self._tile_positions(W)

        print(f"[INFO] Processing image '{KITTI_IMG_PATH}'...")
        print("[INFO] Extracting patch feature maps and stitching global canvas...")

        for y_start, y_fill_s, y_fill_e, y_local_s, y_local_e in y_entries:
            for x_start, x_fill_s, x_fill_e, x_local_s, x_local_e in x_entries:
                patch = full_image[:, :, y_start:y_start + self.tile_dim,
                                          x_start:x_start + self.tile_dim]
                patch_feat = self.patch_stage(patch)

                contributed = patch_feat[:, :, y_local_s:y_local_e, x_local_s:x_local_e]
                stitched_canvas[:, :, y_fill_s:y_fill_e, x_fill_s:x_fill_e] = contributed

                # Save individual mapped feature tensor
                tag = (f"patch_src_y{y_start}_x{x_start}"
                       f"__canvas_y{y_fill_s}-{y_fill_e}_x{x_fill_s}-{x_fill_e}")
                dump_feature_tensor(contributed, tag, out_dir)

        # Save full assembled global canvas feature tensor
        dump_feature_tensor(stitched_canvas, "global_canvas_full", out_dir)
        print(f"[SUCCESS] Saved {len(y_entries) * len(x_entries)} feature patch tensors + 1 complete global canvas feature tensor to '{out_dir}'.")
        return stitched_canvas

# =====================================================================
# 3. EXECUTION
# =====================================================================
if __name__ == "__main__":
    extractor = MCUNetV2_CanvasExtractor(tile_dim=TILE_DIM, overlap=OVERLAP).to(DEVICE)
    
    # 1. Load trained weights from checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
        patch_weights = {k.replace('patch_stage.', '').replace('module.', ''): v 
                         for k, v in state_dict.items() if 'patch_stage.' in k}
        extractor.patch_stage.load_state_dict(patch_weights, strict=True)
        print(f"[INFO] Loaded patch_stage weights from '{CHECKPOINT_PATH}'")
    else:
        raise FileNotFoundError(f"[ERROR] Could not find checkpoint file at '{CHECKPOINT_PATH}'.")

    extractor.eval()
    
    # 2. Load and preprocess 0002/000129.png
    if not os.path.exists(KITTI_IMG_PATH):
        raise FileNotFoundError(f"[ERROR] Image not found at '{KITTI_IMG_PATH}'.")

    img_bgr = cv2.imread(KITTI_IMG_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 192))  # Resize to model input space (640x192)

    # Normalize image tensor [0, 1] and add batch dimension (1, 3, 192, 640)
    real_image_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    real_image_tensor = real_image_tensor.to(DEVICE)
    
    # 3. Extract feature tensors
    with torch.no_grad():
        global_canvas_tensor = extractor.extract_canvas(real_image_tensor, OUTPUT_DIR)