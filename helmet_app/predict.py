"""
predict.py — Helmet Detection (command-line)
=============================================
Give an image as input → get helmet counts as output.

Usage:
    python predict.py image.jpg
    python predict.py path/to/photo.png
    python predict.py image.jpg --conf 0.3
"""

import sys
import os
import glob
import argparse
import pathlib

# ── Parse arguments ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Helmet Detection Counter")
parser.add_argument("image",        help="Path to input image (jpg/png)")
parser.add_argument("--conf", "-c", type=float, default=0.40,
                    help="Confidence threshold 0.0–1.0  (default: 0.40)")
parser.add_argument("--save", "-s", action="store_true",
                    help="Save annotated output image alongside input")
args = parser.parse_args()

# ── Validate image path ───────────────────────────────────────────────────────
img_path = pathlib.Path(args.image).resolve()
if not img_path.exists():
    print(f"[ERROR] Image not found: {img_path}")
    sys.exit(1)

# ── Find best.pt (search from this script's directory upward) ─────────────────
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
search_dirs = [SCRIPT_DIR, SCRIPT_DIR.parent]
hits = []
for d in search_dirs:
    hits += glob.glob(str(d / "**" / "best.pt"), recursive=True)

if not hits:
    print("[ERROR] No trained model (best.pt) found.")
    print("        Run  python train.py  first, then try again.")
    sys.exit(1)

model_path = max(hits, key=os.path.getmtime)
print(f"[INFO] Using model : {model_path}")
print(f"[INFO] Image       : {img_path}")
print(f"[INFO] Confidence  : {args.conf}")
print()

# ── Load model ────────────────────────────────────────────────────────────────
from ultralytics import YOLO
model = YOLO(model_path)

# ── Class sets (covers every label variant stored in best.pt) ─────────────────
# The dataset originally had 0=Helmet, 1=No Helmet (the names in the model are flipped)
HELMET_IDS    = {1, "1", "helmet",    "Helmet"}
NO_HELMET_IDS = {0, "0", "no_helmet", "No Helmet"}

# ── Run inference ─────────────────────────────────────────────────────────────
results = model.predict(
    source  = str(img_path),
    conf    = args.conf,
    iou     = 0.45,
    verbose = False,
)

with_helmet    = 0
without_helmet = 0

for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        raw    = model.names.get(cls_id, str(cls_id))

        if cls_id in HELMET_IDS or raw in HELMET_IDS:
            with_helmet += 1
        elif cls_id in NO_HELMET_IDS or raw in NO_HELMET_IDS:
            without_helmet += 1

# ── Print result ──────────────────────────────────────────────────────────────
total = with_helmet + without_helmet
print("=" * 40)
print(f"  ✅ With Helmet    : {with_helmet}")
print(f"  ❌ Without Helmet : {without_helmet}")
print(f"  👥 Total Detected : {total}")
print("=" * 40)

if total == 0:
    print("  ℹ️  No persons detected. Try a lower --conf value.")
elif without_helmet > 0:
    print(f"  ⚠️  WARNING: {without_helmet} person(s) not wearing a helmet!")
else:
    print("  ✅ SAFE: Everyone is wearing a helmet.")
print()

# ── Save annotated image (optional) ──────────────────────────────────────────
if args.save:
    import cv2
    import numpy as np

    GREEN = (16, 185, 129)
    RED   = (239, 68,  68)

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf_v = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            raw = model.names.get(cls_id, str(cls_id))

            if cls_id in HELMET_IDS or raw in HELMET_IDS:
                label, color = "With Helmet", GREEN
            elif cls_id in NO_HELMET_IDS or raw in NO_HELMET_IDS:
                label, color = "Without Helmet", RED
            else:
                continue

            bgr   = (color[2], color[1], color[0])
            thick = max(2, w // 400)
            cv2.rectangle(img, (x1, y1), (x2, y2), bgr, thick)

            text = f"{label} {conf_v:.0%}"
            fs = max(0.45, w / 2000)
            ft = max(1, thick - 1)
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
            ty1 = max(y1 - th - bl - 6, 0)
            ty2 = ty1 + th + bl + 6
            cv2.rectangle(img, (x1, ty1), (x1 + tw + 8, ty2), bgr, -1)
            cv2.putText(img, text, (x1 + 4, ty2 - bl - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), ft, cv2.LINE_AA)

    out_path = img_path.parent / f"result_{img_path.name}"
    cv2.imwrite(str(out_path), img)
    print(f"  💾 Saved annotated image → {out_path}")
