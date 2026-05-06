"""
🪖 Helmet Detection System
Supports: Image & Video Upload.
Features: 
- Counts persons WITH / WITHOUT helmet.
- Captures screenshots of violations (No Helmet) and stores them in a folder.

Run:
    streamlit run app.py
"""

import io, os, glob, time, pathlib, tempfile, shutil, sys
import cv2, numpy as np
import streamlit as st
import yaml
import torch
from PIL import Image

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🪖 Helmet Detection & Enforcement",
    page_icon="🪖",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0f1117; }

.title  { font-size:2.4rem; font-weight:800; color:#f1f5f9; text-align:center; margin-bottom:.2rem; }
.sub    { text-align:center; color:#64748b; margin-bottom:2rem; }

/* cards */
.card {
    border-radius:16px; padding:1.6rem 1rem;
    text-align:center; border:2px solid;
}
.card-green { background:#052e16; border-color:#16a34a; }
.card-red   { background:#450a0a; border-color:#dc2626; }
.card-val   { font-size:3rem; font-weight:800; }
.card-lbl   { font-size:.9rem; margin-top:.3rem; color:#cbd5e1; }
.green-val  { color:#4ade80; }
.red-val    { color:#f87171; }

/* banners */
.ok-banner   { background:#052e16; border-left:5px solid #16a34a;
               border-radius:10px; padding:1rem 1.4rem; color:#bbf7d0;
               font-weight:600; font-size:1.05rem; margin-top:1rem; }
.warn-banner { background:#450a0a; border-left:5px solid #dc2626;
               border-radius:10px; padding:1rem 1.4rem; color:#fecaca;
               font-weight:600; font-size:1.05rem; margin-top:1rem; }
.info-banner { background:#1e1b4b; border-left:5px solid #818cf8;
               border-radius:10px; padding:1rem 1.4rem; color:#c7d2fe;
               font-size:.95rem; margin-top:1rem; }

/* Violation Gallery */
.violation-title { color: #f87171; font-weight: 700; margin-top: 2rem; border-bottom: 1px solid #450a0a; padding-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# ── Training Logic (Integrated from train_helmet.py) ─────────────────────────
def create_dataset_yaml(dataset_root, yaml_path):
    cfg = {
        "path":  str(dataset_root),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    2,
        "names": ["no_helmet", "helmet"],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return str(yaml_path)

def get_device():
    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def run_training():
    from ultralytics import YOLO
    dataset_root = SCRIPT_DIR.parent / "archive (6)"
    yaml_path = SCRIPT_DIR / "dataset.yaml"
    
    if not dataset_root.exists():
        st.error(f"Dataset not found at {dataset_root}. Please ensure the 'archive (6)' folder exists.")
        return False

    create_dataset_yaml(dataset_root, yaml_path)
    device = get_device()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.info("🚀 Loading base YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    
    status_text.info("🏋️ Training started (this may take a few minutes)...")
    
    results = model.train(
        data      = str(yaml_path),
        epochs    = 5, 
        imgsz     = 640,
        batch     = 16,
        project   = str(SCRIPT_DIR / "helmet_detection"),
        name      = "run1",
        exist_ok  = True,
        device    = device,
        workers   = 0,
        verbose   = True
    )
    
    progress_bar.progress(100)
    status_text.success("✅ Training Complete!")
    return True

# ── Constants ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
VIOLATIONS_DIR = SCRIPT_DIR / "violations"
VIOLATIONS_DIR.mkdir(exist_ok=True)

# Class mapping from the trained model:
HELMET_CLASSES    = {1, "1", "helmet",    "Helmet",    "WITH HELMET"}
NO_HELMET_CLASSES = {0, "0", "no_helmet", "No Helmet", "WITHOUT HELMET"}
GREEN = (16, 185, 129)   # helmet  → green
RED   = (239, 68,  68)   # no helmet → red


# ── Find model ──────────────────────────────────────────────────────────────
def find_model() -> str | None:
    # 1. Look for custom trained model in known locations
    search_patterns = [
        str(SCRIPT_DIR / "helmet_detection" / "run1" / "weights" / "best.pt"),
        str(SCRIPT_DIR / "best.pt"),
        "**/best.pt"
    ]
    
    for pattern in search_patterns:
        if "**" in pattern:
            hits = glob.glob(str(SCRIPT_DIR / pattern), recursive=True)
        else:
            hits = glob.glob(pattern)
        
        if hits:
            return max(hits, key=os.path.getmtime)
    
    # 2. Fallback to base model if it exists in root
    base_model = SCRIPT_DIR / "yolov8n.pt"
    if base_model.exists():
        return str(base_model)
        
    return None


# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str, mtime: float):
    from ultralytics import YOLO
    return YOLO(path)


# ── Detection logic ───────────────────────────────────────────────────────────
def draw_detections(model, img_bgr, results, conf: float, save_violations=False):
    h, w = img_bgr.shape[:2]
    with_helmet = 0
    without_helmet = 0
    
    # Create a copy for saving clean crops if needed
    clean_frame = img_bgr.copy()

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf_v = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            raw = model.names.get(cls_id, str(cls_id))

            if cls_id in HELMET_CLASSES or raw in HELMET_CLASSES:
                label = "With Helmet"
                color = GREEN
                with_helmet += 1
            elif cls_id in NO_HELMET_CLASSES or raw in NO_HELMET_CLASSES:
                label = "Without Helmet"
                color = RED
                without_helmet += 1
                
                # Save violation screenshot
                if save_violations:
                    # Pad the crop slightly
                    px, py = 20, 20
                    vx1, vy1 = max(0, x1-px), max(0, y1-py)
                    vx2, vy2 = min(w, x2+px), min(h, y2+py)
                    crop = clean_frame[vy1:vy2, vx1:vx2]
                    
                    if crop.size > 0:
                        timestamp = time.strftime("%H%M%S")
                        rand_id = np.random.randint(1000, 9999)
                        fname = f"violation_{timestamp}_{rand_id}.jpg"
                        cv2.imwrite(str(VIOLATIONS_DIR / fname), crop)
            else:
                continue

            bgr = (color[2], color[1], color[0])
            thick = max(2, w // 400)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), bgr, thick)

            text = f"{label} {conf_v:.0%}"
            fs = max(0.45, w / 2000)
            ft = max(1, thick - 1)
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
            ty1 = max(y1 - th - bl - 6, 0)
            ty2 = ty1 + th + bl + 6
            cv2.rectangle(img_bgr, (x1, ty1), (x1 + tw + 8, ty2), bgr, -1)
            cv2.putText(img_bgr, text, (x1 + 4, ty2 - bl - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), ft, cv2.LINE_AA)
    
    return img_bgr, with_helmet, without_helmet


def detect_image(model, pil_img: Image.Image, conf: float, save_violations=False):
    img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    t0 = time.perf_counter()
    results = model.predict(source=pil_img, conf=conf, iou=0.45, verbose=False)
    ms = (time.perf_counter() - t0) * 1000
    
    processed_bgr, n_with, n_without = draw_detections(model, img_bgr, results, conf, save_violations)
    out = Image.fromarray(cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB))
    return out, n_with, n_without, ms


# ═════════════════════════════════════════════════════════════════════════════
#  UI
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="title">🪖 Helmet Detection & Enforcement</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">AI-Powered Compliance Monitoring & Violation Capture</p>', unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    option = st.selectbox("Select Input Type", ["Image", "Video"], index=0)
    conf = st.slider("Confidence threshold", 0.05, 1.00, 0.20, 0.05)
    
    st.markdown("---")
    st.markdown("### 📸 Capture Settings")
    auto_capture = st.checkbox("Auto-Capture Violations", value=True, help="Automatically save screenshots of people without helmets")
    always_train = st.checkbox("🔄 Always Train before Inference", value=False, help="Runs a quick training session before every detection")
    
    if st.button("🗑️ Clear All Violations"):
        for f in VIOLATIONS_DIR.glob("*.jpg"):
            os.remove(f)
        st.rerun()

    st.markdown("---")
    st.markdown("### 🛠️ Model Management")
    if st.button("🚀 Train Model Now", help="Always train the model before detection"):
        if run_training():
            st.session_state['trained'] = True
            st.rerun()

# Auto-detect model on startup
model_path = find_model()
if model_path:
    st.session_state['trained'] = True
    mtime = os.path.getmtime(model_path)
    model = load_model(model_path, mtime)
    
    # Show model info in sidebar
    with st.sidebar:
        model_name = os.path.basename(model_path)
        st.success(f"✅ Active Model: `{model_name}`")
else:
    st.session_state['trained'] = False
    with st.sidebar:
        st.warning("⚠️ No model found. Please train or upload `best.pt`.")

# ── Application Logic ────────────────────────────────────────────────────────
if not st.session_state.get('trained', False):
    st.info("👋 **Welcome!** No detection model was found in the project directory.")
    st.markdown("""
    To use this app, you can:
    1. **Train the model** using the button in the sidebar (requires dataset).
    2. **Upload a pre-trained `best.pt`** to the project folder.
    """)
    if st.button("🚀 Start Training Now", type="primary"):
        if run_training():
            st.session_state['trained'] = True
            st.rerun()
    st.stop()

# ── Inference View ──────────────────────────────────────────────────────────
if option == "Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_img, caption="Original Stream", use_container_width=True)
        
        with st.spinner("Processing..."):
            if always_train:
                with st.expander("🛠️ Performing Quick Training...", expanded=True):
                    run_training()
                # Reload model after training
                model_path = find_model()
                mtime = os.path.getmtime(model_path)
                model = load_model(model_path, mtime)
            
            res_img, n_with, n_without, ms = detect_image(model, pil_img, conf, save_violations=auto_capture)
        
        with col2:
            st.image(res_img, caption="Detection Result", use_container_width=True)

        # Result Cards
        st.markdown("### 📊 Real-time Stats")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.markdown(f'<div class="card card-green"><div class="card-val green-val">{n_with}</div><div class="card-lbl">🟢 Safe (With Helmet)</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="card card-red"><div class="card-val red-val">{n_without}</div><div class="card-lbl">🔴 Violation (No Helmet)</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="card" style="border-color:#64748b; background:#1e293b;"><div class="card-val" style="color:#f1f5f9;">{ms:.0f}</div><div class="card-lbl">⏱ Speed (ms)</div></div>', unsafe_allow_html=True)

else:
    uploaded_video = st.file_uploader("Upload a video stream", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()

        vf = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        count_placeholder = st.empty()

        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break
            
            if always_train:
                # For video, training before every frame is impossible, 
                # so we train once at the start of the video.
                with st.spinner("🛠️ Training before video processing..."):
                    run_training()
                # Reload model
                model_path = find_model()
                mtime = os.path.getmtime(model_path)
                model = load_model(model_path, mtime)
                always_train = False # Only train once per video session to keep it usable
                
            results = model.predict(source=frame, conf=conf, iou=0.45, verbose=False)
            processed_frame, n_with, n_without = draw_detections(model, frame, results, conf, save_violations=auto_capture)
            
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, use_container_width=True)
            
            count_placeholder.markdown(f"""
            <div style='display: flex; gap: 20px; justify-content: center; margin-bottom: 20px;'>
                <div style='background: #052e16; padding: 10px 20px; border-radius: 10px; border: 1px solid #16a34a;'>
                    <span style='color: #4ade80; font-size: 1.5rem; font-weight: bold;'>{n_with}</span>
                    <span style='color: #cbd5e1; margin-left: 10px;'>🟢 Safe</span>
                </div>
                <div style='background: #450a0a; padding: 10px 20px; border-radius: 10px; border: 1px solid #dc2626;'>
                    <span style='color: #f87171; font-size: 1.5rem; font-weight: bold;'>{n_without}</span>
                    <span style='color: #cbd5e1; margin-left: 10px;'>🔴 Violations</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        vf.release()
        os.unlink(tfile.name)

# ── Violation Gallery ────────────────────────────────────────────────────────
st.markdown('<h3 class="violation-title">📸 Captured Violations</h3>', unsafe_allow_html=True)
v_files = sorted(list(VIOLATIONS_DIR.glob("*.jpg")), reverse=True)[:12] # Show last 12

if v_files:
    cols = st.columns(4)
    for idx, v_file in enumerate(v_files):
        with cols[idx % 4]:
            st.image(str(v_file), use_container_width=True)
            st.caption(f"Time: {v_file.stem.split('_')[1]}")
else:
    st.info("No violations captured yet. Upload an image/video with people not wearing helmets.")
