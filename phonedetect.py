import streamlit as st
import cv2
import time
import numpy as np
import simpleaudio as sa
from ultralytics import YOLO

# Set Streamlit config
st.set_page_config(page_title="YOLOv8 Mobile Detection", layout="centered")
st.title("📱 Real-time Mobile Phone Detection with YOLOv8")

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Constants
MOBILE_PHONE_CLASS_ID = 67
ALARM_SOUND = "warning-alarm.WAV"
DETECTION_THRESHOLD = 3  # seconds

# Load alarm
wave_obj = sa.WaveObject.from_wave_file(ALARM_SOUND)

# Initialize session state
if "detecting" not in st.session_state:
    st.session_state.detecting = False

# Function to run detection
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

        if phone_detected:
            if phone_detected_start is None:
                phone_detected_start = time.time()
            elif time.time() - phone_detected_start >= DETECTION_THRESHOLD:
                st.error("🚨 Mobile phone detected for 3 seconds!")
                play_obj = wave_obj.play()
                play_obj.wait_done()
                phone_detected_start = None
        else:
            phone_detected_start = None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    cap.release()

# Buttons
if not st.session_state.detecting:
    if st.button("▶ Start Detection", key="start_btn"):
        st.session_state.detecting = True
        detect_mobile()
else:
    if st.button("⏹ Stop Detection", key="stop_btn"):
        st.session_state.detecting = False
        st.success("Detection stopped.")
