import os
import time
from collections import deque
from ultralytics import YOLO
import cv2

# Initialize the webcam
webcam = cv2.VideoCapture(1)

# Load the custom YOLO model
model_path = os.path.join('.', 'runsv8', 'detect', 'train', 'weights', 'best.pt')
model = YOLO(model_path)  # Load a custom model

threshold = 0.5  # Confidence threshold

# Blink, yawn, and long blink tracking
blink_count = 0
yawn_count = 0
long_blink_count = 0

previous_eye_state = "eyes open"
previous_yawn_state = "not yawning"
time_window = 60  # Time window for rate calculation
start_time = time.time()

blink_history = deque(maxlen=time_window)  # Track blink timestamps
yawn_history = deque(maxlen=time_window)  # Track yawn timestamps
long_blink_history = deque(maxlen=time_window)  # Track long blinks

eye_closed_start_time = None  # Track when eyes were closed
yawn_start_time = None  # Track yawning start time

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLO model on the frame
    results = model(frame)[0]

    # Track eye and yawn states
    eye_state = "eyes open"  # Default state
    yawn_state = "not yawning"  # Default state

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            label = results.names[int(class_id)].lower()
            if label in ["eyes closed", "eyes open"]:
                eye_state = label
            elif label in ["yawning", "not yawning"]:
                yawn_state = label

    current_time = time.time()

    # Detect normal blinks (eye open → eye closed → eye open)
    if previous_eye_state == "eyes open" and eye_state == "eyes closed":
        eye_closed_start_time = current_time

    elif previous_eye_state == "eyes closed" and eye_state == "eyes open":
        if eye_closed_start_time:  # If eyes were closed before
            blink_duration = current_time - eye_closed_start_time
            blink_count += 1
            blink_history.append(current_time)  # Store blink timestamp

            if blink_duration >= 1.5:  # If eyes were closed for ≥ 3 seconds
                long_blink_count += 1
                long_blink_history.append(current_time)  # Store long blink

        eye_closed_start_time = None  # Reset closed eye timer

    # **Improved Yawn Detection** (Must last at least 1.5 seconds)
    if previous_yawn_state == "not yawning" and yawn_state == "yawning":
        yawn_start_time = current_time  # Start yawn timer

    elif previous_yawn_state == "yawning" and yawn_state == "not yawning":
        if yawn_start_time:  # If yawn was in progress
            yawn_duration = current_time - yawn_start_time
            if yawn_duration >= 1.5:  # Only count if yawn lasted ≥ 1.5 sec
                yawn_count += 1
                yawn_history.append(current_time)  # Store yawn timestamp
        yawn_start_time = None  # Reset yawn timer

    previous_eye_state = eye_state  # Update last frame state
    previous_yawn_state = yawn_state  # Update last yawn state

    # Remove old blinks/yawns from the time window
    while blink_history and blink_history[0] < current_time - time_window:
        blink_history.popleft()

    while yawn_history and yawn_history[0] < current_time - time_window:
        yawn_history.popleft()

    while long_blink_history and long_blink_history[0] < current_time - time_window:
        long_blink_history.popleft()

    # Calculate rates per minute
    blink_rate = len(blink_history)
    yawn_rate = len(yawn_history)
    long_blink_rate = len(long_blink_history)

    # Display results
    cv2.putText(frame, f"Blinks/min: {blink_rate}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Yawns/min: {yawn_rate}", (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Long Blinks: {long_blink_rate}", (50, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

    # If drowsy condition is met
    if (blink_rate >= 30 and yawn_rate >= 3) or long_blink_rate >= 5:
        cv2.putText(frame, "DROWSY", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

    # Show frame
    cv2.imshow('Driver Monitoring', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

# Release webcam and close windows
webcam.release()
cv2.destroyAllWindows()