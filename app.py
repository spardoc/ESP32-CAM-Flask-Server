
# Author: vlarobbyk
# Modified by: Samuel Pardo y Jairo Salazar
# Version: 1.0
# Date: 2024-10-20
# Description: A simple example to process video captured by the ESP32-XIAO-S3 or ESP32-CAM-MB in Flask.


from flask import Flask, render_template, Response, stream_with_context, request, jsonify

from io import BytesIO
import cv2
import numpy as np
import requests
import time

app = Flask(__name__)

# IP Address
_URL = 'http://192.168.0.239'
# Default Streaming Port
_PORT = '81'
# Default streaming route
_ST = '/stream'
SEP = ':'

# Parameters for movement detection
background = None
MAX_FRAMES = 1000
THRESH = 60
ASSIGN_VALUE = 255
ALPHA = 0.1

# Capture video from the laptop's webcam
camera = cv2.VideoCapture(0)  # 0 is the default camera index (laptop webcam)

# Global variables for noise levels
salt_percentage = 0
pepper_percentage = 0

mask_size = 3

stream_url = ''.join([_URL, SEP, _PORT, _ST])

def video_capture():
    global background
    t = 0
    prev_time = time.time()
    
    while True:
        # Read frame from the webcam
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture video")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detection movement
        if background is None:
            background = gray

        diff = cv2.absdiff(background, gray)
        _, motion_mask = cv2.threshold(diff, THRESH, ASSIGN_VALUE, cv2.THRESH_BINARY)
        
        # Overlay motion mask on original frame
        motion_overlay = cv2.addWeighted(frame, 0.7, cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        
        # Draw FPS on the frame
        cv2.putText(motion_overlay, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', motion_overlay)
        frame_bytes = buffer.tobytes()

        # Yield the frame in a format Flask can stream
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        t += 1
        if t >= MAX_FRAMES:
            break

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_stream")
def video_stream():
    return Response(video_capture(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
    
if __name__ == "__main__":
    try:
        app.run(debug=False)
    finally:
        # Release the camera resource when the server stops
        camera.release()
        cv2.destroyAllWindows()