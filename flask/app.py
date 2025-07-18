from flask import Flask, render_template, Response
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO Model
model_path = os.path.join('.', 'runs11', 'detect', 'train', 'weights', 'best.pt')
model = YOLO(model_path)  # Load custom model

threshold = 0.5  # Detection confidence threshold

def fun_n():
    webcam = cv2.VideoCapture(1)  # Change to 1 if using an external webcam

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        H, W, _ = frame.shape

        # Run YOLO detection
        results = model(frame)[0]

        # Draw bounding boxes and labels
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                # Draw the bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                # Add class label
                label = results.names[int(class_id)].upper()
                cv2.putText(frame, label, (int(x1), int(y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        # Encode the frame as JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    webcam.release()

@app.route('/')
def display():
    return render_template('webapp.html')

@app.route('/video_feed')
def video_feed():
    return Response(fun_n(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

