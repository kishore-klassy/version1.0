import streamlit as st
import cv2
import time
import numpy as np
from ultralytics import YOLO

# Streamlit setup
st.set_page_config(page_title="YOLOv8 Mobile Detection", layout="centered")
st.title("📱 Real-time Mobile Phone Detection with YOLOv8")

# Load model
model = YOLO("yolov8n.pt")

# Constants
MOBILE_PHONE_CLASS_ID = 67
DETECTION_THRESHOLD = 3  # seconds

# Session states
if "detecting" not in st.session_state:
    st.session_state.detecting = False

# Flicker box container
alert_box = st.empty()

# Detection function
def detect_mobile():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    phone_detected_start = None

    while st.session_state.detecting:
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam not accessible")
            break

        frame = cv2.flip(frame, 1)
        results = model(frame, verbose=False)
        phone_detected = False

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id == MOBILE_PHONE_CLASS_ID:
                    phone_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Mobile Phone", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Danger logic
        if phone_detected:
            if phone_detected_start is None:
                phone_detected_start = time.time()
            elif time.time() - phone_detected_start >= DETECTION_THRESHOLD:
                flicker_html = """
                <style>
                .flicker-box {
                    background-color: red;
                    color: white;
                    padding: 50px;
                    text-align: center;
                    font-size: 32px;
                    font-weight: bold;
                    animation: flicker 0.5s infinite alternate;
                }
                @keyframes flicker {
                    from {opacity: 1;}
                    to {opacity: 0.2;}
                }
                </style>
                <div class="flicker-box">🚨 DANGER: Mobile Phone Detected! 🚨</div>
                """
                alert_box.markdown(flicker_html, unsafe_allow_html=True)
        else:
            phone_detected_start = None
            alert_box.empty()

        # Show video
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    cap.release()

# Control buttons
if not st.session_state.detecting:
    if st.button("▶ Start Detection"):
        st.session_state.detecting = True
        detect_mobile()
else:
    if st.button("⏹ Stop Detection"):
        st.session_state.detecting = False
        st.success("Detection stopped.")
