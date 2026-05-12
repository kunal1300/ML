"""
Helmet Detection — YOLOv8 Training Script
==========================================
Dataset : Roboflow helmet-pwtak (875 images, 2 classes)
  • Class 0 → no_helmet  (person NOT wearing a helmet)
  • Class 1 → helmet     (person wearing a helmet)

Usage:
  python train.py

Outputs:
  helmet_detection/run1/weights/best.pt   ← best weights (use in app.py)
  helmet_detection/run1/weights/last.pt   ← last-epoch weights

Tips:
  • GPU detected automatically (CUDA / MPS). Falls back to CPU.
  • On CPU training is very slow (~15 min/epoch). Use Google Colab
    (free T4 GPU) if you don't have a local GPU — see train_colab.py.
  • After training: evaluate with  model.val(data=..., split='test')
"""

import os
import sys
import yaml
import time
import shutil
import pathlib

# ── 1. Validate dependencies ────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("[ERROR] ultralytics / torch not installed.")
    print("Run:  pip install ultralytics torch")
    sys.exit(1)

# ── 2. Paths ────────────────────────────────────────────────────────────────
# Try to find the dataset folder in current dir or parent dir
DATASET_ROOT = SCRIPT_DIR / "archive (6)"
if not DATASET_ROOT.exists():
    DATASET_ROOT = SCRIPT_DIR.parent / "archive (6)"
YAML_PATH    = SCRIPT_DIR / "dataset.yaml"
WEIGHTS_DIR  = SCRIPT_DIR / "helmet_detection" / "run1" / "weights"

# ── 3. (Re)create dataset.yaml with absolute paths ─────────────────────────
def create_dataset_yaml():
    """Write dataset.yaml using resolved absolute paths so training always works
    regardless of the working directory."""
    cfg = {
        "path":  str(DATASET_ROOT),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    2,
        # 0 = no_helmet, 1 = helmet  (verified from label files)
        "names": ["no_helmet", "helmet"],
    }
    with open(YAML_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"[INFO] dataset.yaml written → {YAML_PATH}")
    return str(YAML_PATH)


# ── 4. Device detection ─────────────────────────────────────────────────────
def get_device():
    """Return 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        print(f"[INFO] GPU detected: {gpu} → using CUDA")
        return 0          # Ultralytics expects int for CUDA device index
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[INFO] Apple Silicon GPU detected → using MPS")
        return "mps"
    print("[WARN] No GPU found → training on CPU (slow!)")
    print("       Consider Google Colab for free GPU training.")
    return "cpu"


# ── 5. Training ─────────────────────────────────────────────────────────────
def train():
    yaml_path = create_dataset_yaml()
    device    = get_device()

    # Validate dataset structure before training
    for split in ("train", "valid", "test"):
        img_dir = DATASET_ROOT / split / "images"
        lbl_dir = DATASET_ROOT / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            print(f"[ERROR] Expected directory not found: {img_dir}")
            print("        Update DATASET_ROOT in train.py to point to your dataset folder.")
            sys.exit(1)

    imgs = list((DATASET_ROOT / "train" / "images").glob("*.*"))
    print(f"[INFO] Training images found: {len(imgs)}")
    if len(imgs) == 0:
        print("[ERROR] No training images found. Check DATASET_ROOT path.")
        sys.exit(1)

    # Load YOLOv8n (nano) — fast enough for CPU; switch to 'yolov8s.pt' for more accuracy on GPU
    print("\n[INFO] Loading YOLOv8n base model …")
    model = YOLO("yolov8n.pt")

    print("\n[INFO] Starting training …")
    print("="*60)
    t0 = time.time()

    results = model.train(
        data      = yaml_path,
        epochs    = 80,           # 50 epochs — good balance for this dataset size
        imgsz     = 640,          # input image size
        batch     = 16,           # reduce to 8 if you hit OOM on GPU
        optimizer = "AdamW",      # AdamW converges faster than SGD for small datasets
        lr0       = 0.001,        # initial learning rate
        lrf       = 0.01,         # final lr = lr0 * lrf
        momentum  = 0.937,
        weight_decay = 0.0005,
        warmup_epochs   = 3,
        warmup_momentum = 0.8,
        # ─── Augmentations ─────────────────────────────────────────
        hsv_h   = 0.015,          # hue augmentation
        hsv_s   = 0.7,            # saturation augmentation
        hsv_v   = 0.4,            # value (brightness) augmentation
        degrees = 10.0,           # rotation
        translate = 0.1,          # translation
        scale   = 0.5,            # scale
        shear   = 2.0,            # shear
        flipud  = 0.0,            # vertical flip probability
        fliplr  = 0.5,            # horizontal flip probability
        mosaic  = 1.0,            # mosaic augmentation (combines 4 images)
        mixup   = 0.1,            # mixup augmentation
        copy_paste = 0.0,
        # ─── Early stopping & saving ────────────────────────────────
        patience = 10,            # stop if no mAP improvement for 10 epochs
        save     = True,
        save_period = 10,         # save checkpoint every 10 epochs
        # ─── Output ─────────────────────────────────────────────────
        project  = str(SCRIPT_DIR / "helmet_detection"),
        name     = "run1",
        exist_ok = True,          # overwrite previous run with same name
        device   = device,
        workers  = 0,             # 0 = main thread (avoids Windows DataLoader issues)
        verbose  = True,
    )

    elapsed = time.time() - t0
    print(f"\n[INFO] Training finished in {elapsed/60:.1f} min")

    # ── 6. Evaluate on test set ─────────────────────────────────────────────
    best_pt = SCRIPT_DIR / "helmet_detection" / "run1" / "weights" / "best.pt"
    if best_pt.exists():
        print("\n[INFO] Evaluating best model on test split …")
        best_model = YOLO(str(best_pt))
        test_results = best_model.val(
            data   = yaml_path,
            split  = "test",
            imgsz  = 640,
            device = device,
        )
        print("\n[TEST RESULTS]")
        print(f"  mAP@50    : {test_results.box.map50:.4f}")
        print(f"  mAP@50-95 : {test_results.box.map:.4f}")
        print(f"  Precision : {test_results.box.mp:.4f}")
        print(f"  Recall    : {test_results.box.mr:.4f}")
    else:
        print(f"[WARN] best.pt not found at {best_pt}. Check training logs.")

    print(f"\n✅ Best weights saved → {best_pt}")
    print("   Copy this path into app.py or launch the app — it will detect it automatically.")


# ── 7. Entry-point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
