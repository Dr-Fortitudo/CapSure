from flask import Flask, render_template, Response, request
import cv2
import onnxruntime as ort
import numpy as np

app = Flask(__name__)
session = ort.InferenceSession("best.onnx")
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(0)  # or use video stream source

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Add detection + draw bounding boxes
            # ...
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
