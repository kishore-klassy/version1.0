import cv2
import time
import torch
import simpleaudio as sa
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for better accuracy

# Define class ID for "cell phone" (COCO dataset ID: 67)
MOBILE_PHONE_CLASS_ID = 67

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize phone detection tracking
phone_detected_start = None  # Track first detection time
detection_threshold = 3  # Seconds before triggering alarm

# Load alarm sound
ALARM_SOUND = "warning-alarm.WAV"  # Replace with your sound file path

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Track if mobile phone is detected
    phone_detected = False

    # Process results
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())  # Class ID
            if cls_id == MOBILE_PHONE_CLASS_ID:
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Mobile Phone", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Check if mobile phone is detected for 3 seconds
    if phone_detected:
        if phone_detected_start is None:
            phone_detected_start = time.time()  # Start timer
        elif time.time() - phone_detected_start > detection_threshold:
            print("ALARM: Mobile phone detected for 3 seconds!")

            # Play alarm sound
            wave_obj = sa.WaveObject.from_wave_file(ALARM_SOUND)
            play_obj = wave_obj.play()
            play_obj.wait_done()  # Wait until sound finishes
    else:
        phone_detected_start = None  # Reset timer

    # Show the frame
    cv2.imshow("YOLO Mobile Phone Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
