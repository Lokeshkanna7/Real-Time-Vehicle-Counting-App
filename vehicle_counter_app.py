# vehicle_counter_app.py

from flask import Flask, request, render_template, redirect, url_for, Response
import torch
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
vehicle_counts = {'auto': 0, 'heavy': 0}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def classify_vehicle(label):
    heavy = ['truck', 'bus']
    auto = ['car', 'motorcycle', 'van']
    if label in heavy:
        return 'heavy'
    elif label in auto:
        return 'auto'
    else:
        return 'other'

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    vehicle_counts = {'auto': 0, 'heavy': 0}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 5 != 0:
            continue
        results = model(frame)
        detections = results.pandas().xyxy[0]

        for _, row in detections.iterrows():
            label = row['name']
            vehicle_type = classify_vehicle(label)
            if vehicle_type in vehicle_counts:
                vehicle_counts[vehicle_type] += 1

    cap.release()
    return vehicle_counts

def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        if frame_count % 10 != 0:
            continue

        results = model(frame)
        for *box, conf, cls in results.xyxy[0]:
            label = model.names[int(cls)]
            vehicle_type = classify_vehicle(label)
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if vehicle_type == 'auto' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{vehicle_type} ({label})', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Encode frame as JPEG and stream
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            counts = process_video(filepath)
            return render_template('results.html', counts=counts, filename=filename)
    return render_template('index.html')

@app.route('/video_feed/<filename>')
def video_feed(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(generate_frames(filepath),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
