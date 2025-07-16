🚗 Vehicle Counter Web App using YOLOv5, Flask, and OpenCV

A real-time web application that detects and counts vehicles from uploaded videos using the YOLOv5 object detection model. It classifies vehicles into two categories — `auto` (cars, motorcycles, vans) and `heavy` (buses, trucks) — and provides annotated video streams with bounding boxes.

---

## 🔍 Project Overview

This application allows users to upload a video file through a web interface, processes the video frame-by-frame using the YOLOv5 deep learning model, and returns:

- Real-time vehicle detection
- Vehicle type classification (`auto` vs `heavy`)
- Live annotated video feed
- Summary of total vehicle counts

---

## ✨ Features

- 📤 Upload videos in formats like `.mp4`, `.avi`, `.mov`
- 🧠 Object detection using pretrained **YOLOv5s** model (via PyTorch Hub)
- 🚘 Vehicle classification:
  - `Auto`: car, motorcycle, van
  - `Heavy`: truck, bus
- 🖼️ Annotated video stream with labeled bounding boxes
- 📊 Dashboard showing counts of detected vehicles

---

## 🧰 Tech Stack

| Component     | Technology         |
|---------------|--------------------|
| Backend       | Python, Flask      |
| Frontend      | HTML               |
| Model         | YOLOv5s (Ultralytics) |
| Video Handling| OpenCV             |
| Deployment    | Localhost (Flask server) |

---

## 📁 Project Structure

vehicle_counter_app/
│
├── templates/
│ ├── index.html # Upload page
│ └── results.html # Results + video stream
│
├── uploads/ # Stores uploaded videos
│
├── vehicle_counter_app.py # Main Flask application
└── README.md
