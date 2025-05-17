import time
import streamlit as st
from ultralytics import YOLO
import cv2
import pygame

# Initialize pygame only once
pygame.mixer.init()
pygame.mixer.music.load("warning-alarm.WAV")  # Make sure alarm.mp3 exists in folder

# Load the model only once
model = YOLO("yolov8n.pt")

st.title("Mobile Phone Detection")

# Initialize session state variables
if "detecting" not in st.session_state:
    st.session_state.detecting = False
if "detection_start_time" not in st.session_state:
    st.session_state.detection_start_time = None
if "alarm_playing" not in st.session_state:
    st.session_state.alarm_playing = False

# Buttons to control detection
if st.button("Start Detection"):
    st.session_state.detecting = True
    st.session_state.detection_start_time = None

if st.button("Stop Detection"):
    st.session_state.detecting = False
    st.session_state.detection_start_time = None
    if st.session_state.alarm_playing:
        pygame.mixer.music.stop()
        st.session_state.alarm_playing = False

stframe = st.empty()

if st.session_state.detecting:
    cap = cv2.VideoCapture(0)
    while cap.isOpened() and st.session_state.detecting:
        success, frame = cap.read()
        if not success:
            st.warning("Cannot access webcam.")
            break

        results = model.predict(frame, conf=0.5)
        annotated_frame = results[0].plot()

        detected = False
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.model.names[cls]
            if label == "cell phone":
                detected = True
                break

        if detected:
            if st.session_state.detection_start_time is None:
                st.session_state.detection_start_time = time.time()
            elif time.time() - st.session_state.detection_start_time >= 3:
                if not st.session_state.alarm_playing:
                    pygame.mixer.music.play()
                    st.session_state.alarm_playing = True
        else:
            st.session_state.detection_start_time = None
            if st.session_state.alarm_playing:
                pygame.mixer.music.stop()
                st.session_state.alarm_playing = False

        stframe.image(annotated_frame, channels="BGR")

    cap.release()
else:
    stframe.text("Detection stopped. Click 'Start Detection' to begin.")
