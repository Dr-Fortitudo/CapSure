import os
os.environ["PYTHON_ZONEINFO_TZPATH"] = "tzdata"

import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime
from zoneinfo import ZoneInfo
from PIL import Image
import zipfile
import pandas as pd

# Config
st.set_page_config(page_title="CapSure", page_icon="🪖", layout="wide")

# Constants
MODEL_ZIP = "app/best.zip"
MODEL_PATH = "app/best.onnx"
ALARM_PATH = "app/alarm.mp3"
LOGO_PATH = "app/CapSure_logo.png"
LABELS = ["NO Helmet", "ON. Helmet"]

# Unzip ONNX model if needed
if not os.path.exists(MODEL_PATH):
    with zipfile.ZipFile(MODEL_ZIP, "r") as z:
        z.extractall("app")

# Load ONNX Model
@st.cache_resource
def load_model():
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name

session, input_name = load_model()

# Preprocess
def preprocess(img):
    img_resized = cv2.resize(img, (640, 640))
    img_transposed = img_resized.transpose(2, 0, 1)
    img_normalized = img_transposed.astype(np.float32) / 255.0
    return np.expand_dims(img_normalized, axis=0)

# Postprocess
def postprocess(outputs, threshold=0.3):
    predictions = outputs[0][0]
    results = []
    for pred in predictions:
        if len(pred) < 6:
            continue
        x1, y1, x2, y2, conf, cls = pred[:6]
        if conf > threshold:
            results.append((int(cls), float(conf), (int(x1), int(y1), int(x2), int(y2))))
    return results

# State init
if "history" not in st.session_state:
    st.session_state.history = []
if "violation" not in st.session_state:
    st.session_state.violation = False

# Sidebar UI
st.sidebar.image(LOGO_PATH, use_container_width=True)
st.sidebar.markdown(
    """
    <h1 style='text-align:center; color:yellow;'>CapSure</h1>
    <h3 style='text-align:center; color:lightgray;'>Helmet Compliance Detection</h3>
    """,
    unsafe_allow_html=True
)
start_camera = st.sidebar.toggle("📷 Camera ON/OFF", value=False)
reset_trigger = st.sidebar.button("🔁 RESET")

# Title
st.markdown("<h1 style='text-align:center; color:#3ABEFF;'>CapSure Helmet Detection System</h1>", unsafe_allow_html=True)
st.markdown("---")

frame_placeholder = st.empty()

# Camera Detection Logic
if start_camera and not st.session_state.violation:
    img_file = st.camera_input("Capture Image")
    if img_file:
        image = Image.open(img_file).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        img_input = preprocess(frame)
        outputs = session.run(None, {input_name: img_input})
        detections = postprocess(outputs)

        alert_triggered = False

        for cls_id, conf, (x1, y1, x2, y2) in detections:
            label = LABELS[cls_id]
            color = (0, 255, 0) if label == "ON. Helmet" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if label == "NO Helmet":
                alert_triggered = True

        # Alert Logic
        if alert_triggered:
            st.audio(ALARM_PATH, format="audio/mp3", start_time=0)
            dt = datetime.now(ZoneInfo("Asia/Kolkata"))
            ts = dt.strftime("%I:%M:%S %p @ %d %B, %Y")
            filename = f"violation_{dt.strftime('%Y%m%d_%H%M%S')}.jpg"

            _, img_encoded = cv2.imencode('.jpg', frame)
            img_bytes = img_encoded.tobytes()

            st.session_state.history.insert(0, {
                "timestamp": ts,
                "class": "NO Helmet",
                "filename": filename,
                "image_bytes": img_bytes
            })

            st.session_state.violation = True
            st.warning("🚨 Violation Detected!")
            st.download_button("⬇️ Download Snapshot", img_bytes, filename, "image/jpeg")

        frame_placeholder.image(frame, channels="BGR", use_container_width=True)

elif st.session_state.violation:
    st.info("Detection paused. Press RESET to resume.")

if reset_trigger:
    st.session_state.violation = False
    st.rerun()

# Defect Log
st.markdown("---")
st.markdown("## 📋 Defect Log")
if st.session_state.history:
    df = pd.DataFrame([{"Time": h["timestamp"], "Class": h["class"]} for h in st.session_state.history])
    st.dataframe(df)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Log CSV", csv, "defect_log.csv", "text/csv")
else:
    st.info("No violations yet.")
