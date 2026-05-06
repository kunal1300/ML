# 🪖 Helmet Detection & Enforcement System

This application uses **YOLOv8** to detect people with and without helmets in real-time. It supports both image and video uploads and automatically captures screenshots of violations.

## 🚀 Deployment Instructions

1. **Push to GitHub**: Ensure all files (including `best.pt` or `yolov8n.pt`) are pushed to your repository.
2. **Connect to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/).
   - Connect your GitHub repo.
   - Select `app.py` as the main file.
3. **Wait for Build**: Streamlit will automatically install dependencies from `requirements.txt` and `packages.txt`.

## 📁 Project Structure

- `app.py`: Main Streamlit application.
- `requirements.txt`: Python dependencies.
- `packages.txt`: System-level dependencies (OpenCV/FFmpeg).
- `helmet_detection/`: Contains the trained YOLO model.
- `violations/`: Directory where violation screenshots are saved.

## 🛠️ Features

- **Real-time Stats**: Track safe vs. violation counts.
- **Auto-Capture**: Automatically saves crops of individuals not wearing helmets.
- **Custom UI**: Dark-themed, premium interface with interactive controls.
- **Model Management**: Option to retrain the model if a dataset is provided.

## ⚖️ License
MIT
