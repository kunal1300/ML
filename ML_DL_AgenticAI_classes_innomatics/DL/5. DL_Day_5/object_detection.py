import os
# --- STREAMLIT CLOUD DEPLOYMENT FIX ---
# Streamlit Cloud's Debian server is currently failing to install libglib2.0-0.
# Because Ultralytics forces the full 'opencv-python' installation, it crashes.
# This script forcefully strips out the GUI version and replaces it with headless.
try:
    import cv2
except ImportError:
    os.system("pip uninstall -y opencv-python opencv-python-headless")
    os.system("pip install opencv-python-headless")

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from collections import Counter
import pandas as pd
import io
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# Set page configuration
st.set_page_config(page_title="Advanced Object Detection", layout="wide")
st.title("🚀 Advanced Object Detection Studio")
st.write("Detect, track, and analyze objects using YOLOv8.")

# Sidebar Configuration
st.sidebar.header("1. Model Configuration")

# 5. Swap to a Bigger Model & 10. Switch to Custom Model
model_option = st.sidebar.selectbox("Choose Model", ["yolov8n.pt (Fast)", "yolov8s.pt (Balanced)", "yolov8m.pt (Accurate)", "Custom Model..."])
if model_option == "Custom Model...":
    custom_weights = st.sidebar.file_uploader("Upload custom .pt file", type=["pt"])
    if custom_weights:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
            tmp.write(custom_weights.read())
            model_name = tmp.name
    else:
        st.sidebar.warning("Please upload a .pt file to proceed.")
        st.stop()
else:
    model_name = model_option.split(" ")[0]

@st.cache_resource
def load_model(name):
    return YOLO(name)

model = load_model(model_name)
class_names = list(model.names.values())

# 1. Confidence Threshold Control
st.sidebar.header("2. Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.30, 0.05)

# 3. Custom Class Filtering
selected_classes = st.sidebar.multiselect("Classes to Detect", class_names, default=["person", "car", "cell phone"])
selected_indices = [k for k, v in model.names.items() if v in selected_classes]
if not selected_indices: # Fallback if empty
    selected_indices = list(model.names.keys())

# 8. Alert System
st.sidebar.header("3. Alert System")
alert_target = st.sidebar.selectbox("Trigger Alert On", ["None"] + class_names)

# 9. Zone-Based Detection (Simple ROI)
st.sidebar.header("4. Region of Interest (ROI)")
st.sidebar.write("Filter detections by screen area (%)")
roi_y_min, roi_y_max = st.sidebar.slider("Y-Axis (Top to Bottom)", 0, 100, (0, 100))
roi_x_min, roi_x_max = st.sidebar.slider("X-Axis (Left to Right)", 0, 100, (0, 100))

source = st.sidebar.radio("5. Select Source", ["Image Upload", "Video Upload", "Live Webcam (Local PC)", "Live Webcam (WebRTC/Cloud)", "Mobile Camera Snapshot"])

def process_results(results, frame_shape):
    """Helper to process results, apply ROI, and extract info"""
    detected_classes = []
    data_list = []
    h, w = frame_shape[:2]
    
    ymin, ymax = int(h * roi_y_min / 100), int(h * roi_y_max / 100)
    xmin, xmax = int(w * roi_x_min / 100), int(w * roi_x_max / 100)
    
    img_with_boxes = results[0].orig_img.copy()
    
    if not (roi_y_min == 0 and roi_y_max == 100 and roi_x_min == 0 and roi_x_max == 100):
        cv2.rectangle(img_with_boxes, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
        cv2.putText(img_with_boxes, "ROI", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    for box in results[0].boxes:
        bx1, by1, bx2, by2 = [int(i) for i in box.xyxy[0]]
        
        # Tracking IDs (if available)
        track_id = int(box.id[0]) if box.id is not None else None
        
        cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
        if xmin <= cx <= xmax and ymin <= cy <= ymax:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = float(box.conf[0])
            
            detected_classes.append(class_name)
            data_list.append({
                "Object": class_name,
                "Confidence": conf,
                "Bounding Box": [bx1, by1, bx2, by2],
                "Track ID": track_id
            })
            
            # Generate bright, distinct color for the class
            import colorsys
            hue = (class_id * 137.508) % 360 / 360.0
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            color = (int(r * 255), int(g * 255), int(b * 255))
            
            label = f"{class_name} {conf:.2f}"
            if track_id: label += f" ID:{track_id}"
            
            cv2.rectangle(img_with_boxes, (bx1, by1), (bx2, by2), color, 2)
            cv2.putText(img_with_boxes, label, (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    # Draw object counts on the top-left of the frame
    class_counts = Counter(detected_classes)
    y_offset = 30
    for obj, count in class_counts.items():
        # Find class_id to match the color
        class_id_for_color = list(model.names.keys())[list(model.names.values()).index(obj)]
        import colorsys
        hue = (class_id_for_color * 137.508) % 360 / 360.0
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        color = (int(r * 255), int(g * 255), int(b * 255))
        
        # Add background rectangle for better text visibility
        text = f"{obj.capitalize()}: {count}"
        (t_w, t_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img_with_boxes, (10, y_offset - t_h - 5), (10 + t_w, y_offset + 5), (0, 0, 0), -1)
        cv2.putText(img_with_boxes, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y_offset += 35
            
    return img_with_boxes, detected_classes, pd.DataFrame(data_list)

if source == "Image Upload":
    st.header("📸 Image Processing & Export")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
            
        if st.button("Detect Objects"):
            with st.spinner("Processing..."):
                img_array = np.array(image.convert("RGB"))
                
                results = model(img_array, conf=conf_threshold, classes=selected_indices)
                img_with_boxes, detected_classes, df = process_results(results, img_array.shape)
                
                if alert_target in detected_classes:
                    st.error(f"🚨 ALERT: {alert_target} detected!")
                
                with col2:
                    st.image(img_with_boxes, caption="Result", use_container_width=True)
                    
                st.subheader("📊 Analytics")
                st.write(dict(Counter(detected_classes)))
                
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    if not df.empty:
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Detections (CSV)", csv, "detections.csv", "text/csv")
                with col_dl2:
                    img_pil = Image.fromarray(img_with_boxes)
                    buf = io.BytesIO()
                    img_pil.save(buf, format="PNG")
                    st.download_button("🖼️ Download Annotated Image", buf.getvalue(), "result.png", "image/png")

elif source == "Video Upload":
    st.header("🎞️ Video Processing")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        
        if st.button("Process Video"):
            cap = cv2.VideoCapture(tfile.name)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                FRAME_WINDOW = st.image([])
            with col2:
                st.subheader("Live Counts")
                stats_placeholder = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame, conf=conf_threshold, classes=selected_indices)
                img_with_boxes, detected_classes, _ = process_results(results, frame.shape)
                
                FRAME_WINDOW.image(img_with_boxes)
                
                class_counts = Counter(detected_classes)
                with stats_placeholder.container():
                    if not class_counts:
                        st.info("No objects detected...")
                    else:
                        for obj, count in class_counts.items():
                            st.success(f"**{obj.capitalize()}**: {count}")
                
            cap.release()
            st.success("Video processing complete!")

elif source == "Live Webcam (WebRTC/Cloud)":
    st.header("🌐 Cloud/Mobile Webcam Tracking")
    st.write("This uses WebRTC to access your phone or computer's camera directly in the browser!")
    
    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun.twilio.com:3478"]}
        ]
    })
    
    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            
            # Object Tracking
            results = model.track(img, conf=conf_threshold, classes=selected_indices, persist=True)
            img_with_boxes, detected_classes, _ = process_results(results, img.shape)
            
            return av.VideoFrame.from_ndarray(img_with_boxes, format="bgr24")
        except Exception as e:
            import cv2
            img = frame.to_ndarray(format="bgr24")
            cv2.putText(img, f"Error: {str(e)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
    webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
        async_processing=True,
    )
    
    st.info("💡 Note: The live counts of detected objects are drawn directly onto your video feed so they stay in perfect sync on mobile devices!")

elif source == "Mobile Camera Snapshot":
    st.header("📱 Mobile Camera Snapshot")
    st.write("If strict firewalls or mobile carriers are blocking the live WebRTC video stream, use this mode to instantly capture and process a live photo directly from your camera!")
    
    img_file_buffer = st.camera_input("Take a picture")
    
    if img_file_buffer is not None:
        with st.spinner("Processing..."):
            image = Image.open(img_file_buffer)
            img_array = np.array(image)
            
            results = model(img_array, conf=conf_threshold, classes=selected_indices)
            img_with_boxes, detected_classes, df = process_results(results, img_array.shape)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.image(img_with_boxes, caption="Detection Result", use_container_width=True)
            with col2:
                st.subheader("Detected")
                class_counts = Counter(detected_classes)
                if not class_counts:
                    st.info("No objects detected...")
                else:
                    for obj, count in class_counts.items():
                        st.success(f"**{obj.capitalize()}**: {count}")

elif source == "Live Webcam (Local PC)":
    st.header("💻 Local PC Webcam Tracking")
    st.write("This uses standard OpenCV to access your computer's built-in webcam. (Note: This only works when running locally on your laptop!)")
    run = st.checkbox("Start Local Webcam")
    
    col1, col2 = st.columns([3, 1])
    with col1: 
        FRAME_WINDOW = st.image([])
    with col2: 
        stats_placeholder = st.empty()
        alert_placeholder = st.empty()
        
    st.subheader("📈 Detection History (Total Objects Over Time)")
    chart_placeholder = st.empty()
    history = []
    
    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("🚨 Error: Could not access your physical webcam. \n\n1. Check if your laptop has a webcam.\n2. Ensure it is not currently being used by Zoom/Teams.\n3. Check Windows Settings > Privacy > Camera to allow desktop apps access.")
        else:
            while run:
                ret, frame = cap.read()
                if not ret: 
                    st.error("Error: Failed to read frame from camera. Connection lost.")
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Object Tracking
            results = model.track(frame, conf=conf_threshold, classes=selected_indices, persist=True)
            img_with_boxes, detected_classes, _ = process_results(results, frame.shape)
            
            FRAME_WINDOW.image(img_with_boxes)
            
            class_counts = Counter(detected_classes)
            with stats_placeholder.container():
                if not class_counts:
                    st.info("No objects detected...")
                else:
                    for obj, count in class_counts.items():
                        st.success(f"**{obj.capitalize()}**: {count}")
            
            history.append(len(detected_classes))
            if len(history) > 100: history.pop(0)
            chart_placeholder.line_chart(history)
            
            if alert_target in detected_classes:
                alert_placeholder.error(f"🚨 {alert_target.upper()} DETECTED!")
            else:
                alert_placeholder.empty()
                
        cap.release()