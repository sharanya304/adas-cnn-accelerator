import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# =====================================================================
# 1. READ COMMAND ARGUMENTS DIRECTLY FROM TERMINAL
# =====================================================================
DATASET_DIR = r"D:\CNN_python_training\training"
EPOCHS = 40
TILE_DIM = 128
OVERLAP = 16
LR = 0.0005
CONF_THRESH = 0.5
OUTPUT_DIR = "./train_mcunetv2_output"
NUM_PREVIEWS = -1  # -1 = all test images

for i in range(len(sys.argv)):
    if sys.argv[i] == "--dataset_dir" and i + 1 < len(sys.argv):
        DATASET_DIR = sys.argv[i + 1]
    elif sys.argv[i] == "--epochs" and i + 1 < len(sys.argv):
        EPOCHS = int(sys.argv[i + 1])
    elif sys.argv[i] == "--tile_dim" and i + 1 < len(sys.argv):
        TILE_DIM = int(sys.argv[i + 1])
    elif sys.argv[i] == "--overlap" and i + 1 < len(sys.argv):
        OVERLAP = int(sys.argv[i + 1])
    elif sys.argv[i] == "--lr" and i + 1 < len(sys.argv):
        LR = float(sys.argv[i + 1])
    elif sys.argv[i] == "--thresh" and i + 1 < len(sys.argv):
        CONF_THRESH = float(sys.argv[i + 1])
    elif sys.argv[i] == "--output_dir" and i + 1 < len(sys.argv):
        OUTPUT_DIR = sys.argv[i + 1]
    elif sys.argv[i] == "--num_previews" and i + 1 < len(sys.argv):
        NUM_PREVIEWS = int(sys.argv[i + 1])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Execution Device: {DEVICE}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
PREVIEW_DIR = os.path.join(OUTPUT_DIR, "preview_detections")
os.makedirs(PREVIEW_DIR, exist_ok=True)

CLASS_MAPPING = {"Car": 0, "Van": 0, "Truck": 0, "Cyclist": 1, "Pedestrian": 2, "Person_sitting": 2}
CLASS_NAMES = {0: "Car", 1: "Cyclist", 2: "Pedestrian"}
CLASS_COLORS = {0: (0, 0, 255), 1: (0, 255, 255), 2: (0, 255, 0)}  # BGR

# FIX (box size accuracy): boxes are no longer drawn at a fixed size per
# class. The network now regresses actual (w, h) per detection (see the
# wh_head branch below). FALLBACK_BOX_SIZES is only used as a floor/ceiling
# sanity clamp on the regressed size (guards against a degenerate near-zero
# or absurdly large prediction early in training), not as the box itself.
FALLBACK_BOX_SIZES = {
    0: (80, 50),   # Car
    1: (45, 55),   # Cyclist
    2: (35, 60)    # Pedestrian
}
MIN_BOX_SIZE = 4       # px, in the 640x192 resized-image space
WH_LOSS_WEIGHT = 0.1   # standard CenterNet weighting for the size-regression term

# =====================================================================
# 2. DATASET LOADER
# ---------------------------------------------------------------------
# FIX (Bug #1): the previous version matched images to labels by
# filename stem, assuming one label file per image. Your actual dataset
# is KITTI TRACKING format: one label file per whole SEQUENCE
# (label_02/0000.txt), containing a frame-number column, while images
# are named per-frame (image_02/0000/000000.png). Stems never matched,
# so every image got zero boxes. This version parses each sequence's
# label file once (frame -> boxes), then looks up each image's boxes by
# its (sequence, frame) pair - and reads the correct columns
# (parts[2]=class, parts[6:10]=bbox), not parts[0]/parts[4:8].
# =====================================================================
class KITTIMCUNetV2Dataset(Dataset):
    def __init__(self, base_dir, img_size=(640, 192), ds_factor=32):
        self.base_dir = base_dir
        self.img_w, self.img_h = img_size
        self.ds_factor = ds_factor
        self.grid_w, self.grid_h = self.img_w // ds_factor, self.img_h // ds_factor

        img_root = os.path.join(base_dir, "image_02")
        label_root = os.path.join(base_dir, "label_02")

        # Parse each sequence's label file ONCE: frame_idx -> [(class_id, [l,t,r,b]), ...]
        # Tracking-format columns: frame track_id type truncated occluded alpha
        #                          bbox_left bbox_top bbox_right bbox_bottom ...
        seq_labels = {}
        if os.path.isdir(label_root):
            for label_file in sorted(os.listdir(label_root)):
                if not label_file.lower().endswith(".txt"):
                    continue
                seq_name = os.path.splitext(label_file)[0]
                frame_boxes = {}
                with open(os.path.join(label_root, label_file), "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 10:
                            continue
                        frame_idx = int(parts[0])
                        cls_name = parts[2]
                        if cls_name not in CLASS_MAPPING:
                            continue
                        l, t, r, b = map(float, parts[6:10])
                        frame_boxes.setdefault(frame_idx, []).append(
                            (CLASS_MAPPING[cls_name], [l, t, r, b])
                        )
                seq_labels[seq_name] = frame_boxes
        else:
            print(f"WARNING: label_02 folder not found at '{label_root}' - "
                  f"all images will have empty targets.")

        self.samples = []
        valid_img_exts = (".png", ".jpg", ".jpeg")
        if os.path.isdir(img_root):
            for seq_name in sorted(os.listdir(img_root)):
                seq_img_dir = os.path.join(img_root, seq_name)
                if not os.path.isdir(seq_img_dir):
                    continue
                frame_boxes = seq_labels.get(seq_name, {})
                for img_name in sorted(os.listdir(seq_img_dir)):
                    if not img_name.lower().endswith(valid_img_exts):
                        continue
                    frame_idx = int(os.path.splitext(img_name)[0])
                    entries = frame_boxes.get(frame_idx, [])
                    boxes = np.array([e[1] for e in entries], dtype=np.float32) if entries else np.zeros((0, 4), dtype=np.float32)
                    labels = np.array([e[0] for e in entries], dtype=np.int64) if entries else np.zeros((0,), dtype=np.int64)
                    self.samples.append((os.path.join(seq_img_dir, img_name), boxes, labels))

        if len(self.samples) == 0:
            raise FileNotFoundError(f"ERROR: No images found under '{img_root}'. Check path!")

        n_with_boxes = sum(1 for _, b, _ in self.samples if len(b) > 0)
        print(f"Indexed {len(self.samples)} total images for MCUNetV2 pipeline "
              f"({n_with_boxes} contain at least one labeled box - "
              f"should be the large majority, not zero).")

    def __len__(self):
        return len(self.samples)

    def draw_gaussian(self, heatmap, center, radius=2):
        x, y = int(center[0]), int(center[1])
        height, width = heatmap.shape
        if x < 0 or x >= width or y < 0 or y >= height:
            return
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    val = np.exp(-(dx**2 + dy**2) / (2 * (radius / 2.0)**2))
                    heatmap[ny, nx] = max(heatmap[ny, nx], val)

    def __getitem__(self, idx):
        img_path, boxes, labels = self.samples[idx]

        img_bgr = cv2.imread(img_path)
        orig_h, orig_w, _ = img_bgr.shape
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.img_w, self.img_h))

        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        target_heatmap = np.zeros((3, self.grid_h, self.grid_w), dtype=np.float32)

        # FIX (box size accuracy): alongside the center heatmap, store the
        # actual box (w, h) - in the resized 640x192 image's pixel space -
        # at the grid cell of each object's center, plus a mask marking
        # which grid cells hold a real object. This is what lets the
        # network learn per-object box size instead of a fixed class default.
        target_wh = np.zeros((2, self.grid_h, self.grid_w), dtype=np.float32)
        target_mask = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        for box, cls_id in zip(boxes, labels):
            x_scale = self.img_w / orig_w
            y_scale = self.img_h / orig_h
            cx = ((box[0] + box[2]) / 2.0) * x_scale / self.ds_factor
            cy = ((box[1] + box[3]) / 2.0) * y_scale / self.ds_factor
            self.draw_gaussian(target_heatmap[int(cls_id)], (cx, cy), radius=2)

            gx, gy = int(cx), int(cy)
            if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                w_resized = (box[2] - box[0]) * x_scale
                h_resized = (box[3] - box[1]) * y_scale
                target_wh[0, gy, gx] = w_resized
                target_wh[1, gy, gx] = h_resized
                target_mask[gy, gx] = 1.0

        return (img_tensor, torch.from_numpy(target_heatmap),
                torch.from_numpy(target_wh), torch.from_numpy(target_mask),
                img_bgr, img_path)

# =====================================================================
# 3. MCUNetV2 ARCHITECTURE
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


def estimate_receptive_field(module_list):
    """Rough receptive-field / total-stride estimate for a feedforward stack
    of Conv2d layers (ignores residual-branch effects - fine as an estimate).
    Used only to sanity-check that OVERLAP is large enough for this
    architecture, instead of trusting an arbitrarily chosen number."""
    rf, jump = 1, 1
    for m in module_list:
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size[0]
            s = m.stride[0]
            rf += (k - 1) * jump
            jump *= s
        elif isinstance(m, InvertedResidualBlock):
            for sub in m.conv:
                if isinstance(sub, nn.Conv2d):
                    k = sub.kernel_size[0]
                    s = sub.stride[0]
                    rf += (k - 1) * jump
                    jump *= s
    return rf, jump


class MCUNetV2_Detector(nn.Module):
    def __init__(self, tile_dim=128, overlap=16, num_classes=3):
        super(MCUNetV2_Detector, self).__init__()
        self.tile_dim = tile_dim
        self.overlap = overlap
        self.stride = tile_dim - overlap
        self.patch_ds_factor = 8  # total stride of patch_stage: 2 * 2 * 1 * 2

        self.patch_stage = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            InvertedResidualBlock(16, 24, stride=2, expand_ratio=1, kernel_size=3),
            InvertedResidualBlock(24, 32, stride=1, expand_ratio=3, kernel_size=1),
            InvertedResidualBlock(32, 40, stride=2, expand_ratio=3, kernel_size=3),
        )

        self.layer_stage = nn.Sequential(
            InvertedResidualBlock(40, 64, stride=2, expand_ratio=4, kernel_size=5),
            InvertedResidualBlock(64, 96, stride=1, expand_ratio=6, kernel_size=7),
            InvertedResidualBlock(96, 160, stride=2, expand_ratio=6, kernel_size=5),
            nn.Conv2d(160, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True)
        )

        # FIX (box size accuracy): the old single detection_head only ever
        # produced a per-class center heatmap - there was nothing in the
        # architecture that could output a box size, which is why sizes had
        # to be hardcoded (DEFAULT_BOX_SIZES) at decode time. This splits
        # into a shared trunk plus two plain conv heads: cls_head (center
        # heatmap, unchanged) and wh_head (regresses box width/height at
        # each grid cell). Both heads are ordinary 1x1/3x3 convs - no
        # dynamic control flow or exotic ops - so this stays straightforward
        # to port to an FPGA/ASIC dataflow alongside the rest of the network.
        self.head_trunk = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
        self.wh_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.ReLU(inplace=True)  # box w/h are non-negative
        )

        # ---------------------------------------------------------------
        # FIX (Bug #4, informational): OVERLAP was previously an arbitrary
        # config number with no link to what the network actually needs.
        # This estimates patch_stage's real receptive field and warns if
        # OVERLAP is too small - it does NOT silently change your value.
        # ---------------------------------------------------------------
        rf, jump = estimate_receptive_field(self.patch_stage)
        recommended_min_overlap = max(0, rf - jump)
        print(f"[patch_stage] estimated receptive field ~{rf}px, total stride {jump}x "
              f"-> recommended minimum OVERLAP ~{recommended_min_overlap}px "
              f"(current OVERLAP={overlap}px)")
        if overlap < recommended_min_overlap:
            print(f"  WARNING: OVERLAP={overlap} is smaller than the estimated "
                  f"requirement (~{recommended_min_overlap}px) - patch boundaries "
                  f"may show artifacts. Consider increasing --overlap.")

    def _tile_positions(self, total_size):
        """
        FIX (Bug #3): always anchors a final tile to the true edge, so the
        previous version's dropped bottom/right strip (anything past the
        last exact multiple of `stride`) is now fully covered.

        FIX (Bug #2): returns, for every tile, exactly which portion of the
        canvas it is responsible for filling (fill_start:fill_end) and
        which portion of THAT tile's own output corresponds to it
        (local_start:local_end) - so overlapping tiles never write the same
        canvas pixel twice. Plain assignment replaces the old `+=`.
        """
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

    def forward(self, full_image):
        B, C, H, W = full_image.shape
        device = full_image.device
        ds = self.patch_ds_factor

        canvas_h, canvas_w = H // ds, W // ds
        stitched_canvas = torch.zeros((B, 40, canvas_h, canvas_w), device=device)

        y_entries = self._tile_positions(H)
        x_entries = self._tile_positions(W)

        for y_start, y_fill_s, y_fill_e, y_local_s, y_local_e in y_entries:
            for x_start, x_fill_s, x_fill_e, x_local_s, x_local_e in x_entries:
                patch = full_image[:, :, y_start:y_start + self.tile_dim,
                                          x_start:x_start + self.tile_dim]
                patch_feat = self.patch_stage(patch)

                stitched_canvas[:, :, y_fill_s:y_fill_e, x_fill_s:x_fill_e] = \
                    patch_feat[:, :, y_local_s:y_local_e, x_local_s:x_local_e]

        global_features = self.layer_stage(stitched_canvas)
        trunk_feat = self.head_trunk(global_features)
        heatmap = self.cls_head(trunk_feat)
        wh = self.wh_head(trunk_feat)
        return heatmap, wh

# =====================================================================
# 4. OFFICIAL CENTERNET FOCAL LOSS FORMULATION (unchanged - not buggy;
#    it was only ever being fed empty targets due to Bug #1)
# =====================================================================
class OfficialCenterNetFocalLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=4.0):
        super(OfficialCenterNetFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        pred = torch.clamp(pred, 1e-4, 1.0 - 1e-4)
        pos_inds = target.eq(1.0).float()
        neg_inds = target.lt(1.0).float()

        neg_weights = torch.pow(1.0 - target, self.beta)

        pos_loss = torch.log(pred) * torch.pow(1.0 - pred, self.alpha) * pos_inds
        neg_loss = torch.log(1.0 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        return loss

class WHRegressionLoss(nn.Module):
    """Masked L1 loss on box width/height, only at grid cells that hold a
    real object center (target_mask == 1). Cells with no object contribute
    nothing, since there's no ground-truth size to regress toward there."""
    def __init__(self):
        super(WHRegressionLoss, self).__init__()

    def forward(self, pred_wh, target_wh, mask):
        mask = mask.unsqueeze(1)  # (B, 1, H, W) -> broadcasts over the 2 wh channels
        num_pos = mask.sum().clamp(min=1.0)
        loss = torch.abs(pred_wh - target_wh) * mask
        return loss.sum() / num_pos


def custom_collate(batch):
    imgs = torch.stack([item[0] for item in batch], dim=0)
    target_maps = torch.stack([item[1] for item in batch], dim=0)
    target_whs = torch.stack([item[2] for item in batch], dim=0)
    target_masks = torch.stack([item[3] for item in batch], dim=0)
    raw_bgrs = [item[4] for item in batch]
    img_paths = [item[5] for item in batch]
    return imgs, target_maps, target_whs, target_masks, raw_bgrs, img_paths

full_dataset = KITTIMCUNetV2Dataset(DATASET_DIR, img_size=(640, 192), ds_factor=32)
train_size = int(0.7 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

model = MCUNetV2_Detector(tile_dim=TILE_DIM, overlap=OVERLAP, num_classes=3).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
focal_criterion = OfficialCenterNetFocalLoss()
wh_criterion = WHRegressionLoss()

print("\n" + "="*60)
print(f"STARTING MCUNetV2 TRAINING FOR {EPOCHS} EPOCHS")
print(f"   - Dataset Path: {DATASET_DIR}")
print(f"   - Split: {train_size} Train Images | {test_size} Test Images")
print(f"   - Learning Rate: {LR}")
print(f"   - WH Loss Weight: {WH_LOSS_WEIGHT}")
print("="*60 + "\n")

# Training Loop
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_focal_loss = 0.0
    running_wh_loss = 0.0
    batches = 0

    for imgs, target_maps, target_whs, target_masks, _, _ in train_loader:
        imgs = imgs.to(DEVICE)
        target_maps = target_maps.to(DEVICE)
        target_whs = target_whs.to(DEVICE)
        target_masks = target_masks.to(DEVICE)

        optimizer.zero_grad()
        pred_maps, pred_whs = model(imgs)

        focal_loss = focal_criterion(pred_maps, target_maps)
        wh_loss = wh_criterion(pred_whs, target_whs, target_masks)
        loss = focal_loss + WH_LOSS_WEIGHT * wh_loss
        loss.backward()
        optimizer.step()

        running_focal_loss += focal_loss.item()
        running_wh_loss += wh_loss.item()
        batches += 1

    epoch_focal = running_focal_loss / max(1, batches)
    epoch_wh = running_wh_loss / max(1, batches)
    print(f"[Epoch {epoch:02d}/{EPOCHS:02d}] Focal Loss: {epoch_focal:.4f} | "
          f"WH L1 Loss: {epoch_wh:.4f} | Combined: {epoch_focal + WH_LOSS_WEIGHT*epoch_wh:.4f}")

weights_path = os.path.join(OUTPUT_DIR, f"mcunetv2_{EPOCHS}epochs.pth")
torch.save(model.state_dict(), weights_path)
print(f"\nSaved Model Weights to: '{weights_path}'")

# =====================================================================
# 5. PEAK FINDING & BOUNDING BOX PREVIEWS
# =====================================================================
def extract_bboxes_from_heatmap(pred_map, pred_wh, thresh=0.12):
    # FIX (box size accuracy): w/h now come from the model's own wh_head
    # prediction at each detected center, not a fixed per-class lookup.
    # FALLBACK_BOX_SIZES / MIN_BOX_SIZE only guard against a degenerate
    # (near-zero or NaN) regressed size, e.g. very early in training.
    pad = (3 - 1) // 2
    hmax = nn.functional.max_pool2d(pred_map, (3, 3), stride=1, padding=pad)
    keep = (pred_map == hmax).float() * (pred_map >= thresh).float()

    wh_np = pred_wh[0].cpu().numpy()  # (2, grid_h, grid_w)

    detections = []
    for c in range(pred_map.shape[1]):
        cls_map = (pred_map[0, c] * keep[0, c]).cpu().numpy()
        ys, xs = np.where(cls_map > 0)
        for y, x in zip(ys, xs):
            score = cls_map[y, x]
            cx = int((x + 0.5) * 32)
            cy = int((y + 0.5) * 32)

            w, h = wh_np[0, y, x], wh_np[1, y, x]
            fallback_w, fallback_h = FALLBACK_BOX_SIZES.get(c, (64, 48))
            if not np.isfinite(w) or w < MIN_BOX_SIZE:
                w = fallback_w
            if not np.isfinite(h) or h < MIN_BOX_SIZE:
                h = fallback_h
            w, h = int(round(w)), int(round(h))

            x1, y1 = max(0, cx - w // 2), max(0, cy - h // 2)
            x2, y2 = min(640, cx + w // 2), min(192, cy + h // 2)
            detections.append(([x1, y1, x2, y2], c, score))
    return detections

preview_limit = len(test_dataset) if NUM_PREVIEWS < 0 else min(NUM_PREVIEWS, len(test_dataset))
print(f"\nGENERATING DETECTED BOUNDING BOX PREVIEWS FOR {preview_limit} TEST IMAGES...")
model.eval()
saved_previews = 0

with torch.no_grad():
    for imgs, _, _, _, raw_bgrs, img_paths in test_loader:
        if saved_previews >= preview_limit:
            break

        imgs = imgs.to(DEVICE)
        pred_map, pred_wh = model(imgs)

        raw_bgr = raw_bgrs[0]
        raw_resized = cv2.resize(raw_bgr, (640, 192))

        detections = extract_bboxes_from_heatmap(pred_map, pred_wh, thresh=CONF_THRESH)

        for (x1, y1, x2, y2), cls_id, score in detections:
            color = CLASS_COLORS.get(cls_id, (0, 255, 0))
            label_text = f"{CLASS_NAMES[cls_id]}:{score:.2f}"
            cv2.rectangle(raw_resized, (x1, y1), (x2, y2), color, 2)
            cv2.putText(raw_resized, label_text, (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        file_stem = os.path.splitext(os.path.basename(img_paths[0]))[0]
        save_path = os.path.join(PREVIEW_DIR, f"preview_{saved_previews+1}_{file_stem}_bbox.png")
        cv2.imwrite(save_path, raw_resized)

        saved_previews += 1
        if saved_previews % 100 == 0 or saved_previews == preview_limit:
            print(f"   Saved Detection Preview {saved_previews}/{preview_limit}: '{save_path}'")

print(f"\nALL {saved_previews} DETECTION PREVIEWS SAVED TO: '{PREVIEW_DIR}'")