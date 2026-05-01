import os
import re
import cv2
import time
import base64
import sqlite3
import subprocess
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO
import gdown

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
SNAPSHOT_FOLDER = "static/snapshots"
DB_NAME = "detections.db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

# 🔥 رابط الموديل
MODEL_URL = "https://drive.google.com/uc?id=1Bd0EvtNsagapzoDQ1zMPKePceyjlJ6oJ"

if not os.path.exists("best.pt"):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, "best.pt", quiet=False)

model = YOLO("best.pt")

last_saved_time_live = 0


# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        object_type TEXT,
        confidence REAL,
        detected_at TEXT,
        image_path TEXT,
        source TEXT
    )
    """)

    conn.commit()
    conn.close()


def insert_detection(obj, conf, time_str, img_path, source):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO detections (object_type, confidence, detected_at, image_path, source)
    VALUES (?, ?, ?, ?, ?)
    """, (obj, conf, time_str, img_path, source))

    conn.commit()
    conn.close()


def get_logs():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM detections ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows


# ================= DRAW =================
def draw_boxes(frame, results):
    annotated = frame.copy()
    detections = []

    if results[0].boxes is None:
        return annotated, detections

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = str(model.names[cls_id]).lower()

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if name == "drone":
            color = (0, 0, 255)
        elif name == "bird":
            color = (0, 255, 0)
        else:
            continue

        label = f"{name} {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        detections.append((name, conf))

    return annotated, detections


init_db()


# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return "No video uploaded", 400

    file = request.files["video"]
    if file.filename == "":
        return "No selected video", 400

    safe_name = re.sub(r"\s+", "_", file.filename)
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "", safe_name)

    filename = f"{int(time.time())}_{safe_name}"
    input_path = os.path.join(UPLOAD_FOLDER, filename)

    temp_output = os.path.join(RESULT_FOLDER, f"temp_{filename}.avi")
    final_output = os.path.join(RESULT_FOLDER, f"out_{filename}.mp4")

    file.save(input_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return "Video error"

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(5)
    if fps <= 0:
        fps = 25

    out = cv2.VideoWriter(
        temp_output,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (width, height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.3, imgsz=320, verbose=False)
        annotated, _ = draw_boxes(frame, results)
        out.write(annotated)

    cap.release()
    out.release()

    # 🔥 تحويل الفيديو (أهم جزء)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", temp_output,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        final_output
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    print("FFMPEG:", result.stderr)

    if result.returncode != 0:
        return f"FFmpeg error:\n{result.stderr}"

    os.remove(temp_output)

    return render_template(
        "index.html",
        uploaded_video=input_path.replace("static/", ""),
        result_video=final_output.replace("static/", "")
    )


# ================= LIVE =================
@app.route("/process_frame", methods=["POST"])
def process_frame():
    global last_saved_time_live

    data = request.get_json()
    image_data = data["image"]

    image_bytes = base64.b64decode(image_data.split(",")[1])
    np_img = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    results = model(frame, conf=0.3, imgsz=320, verbose=False)
    annotated, _ = draw_boxes(frame, results)

    _, buffer = cv2.imencode(".jpg", annotated)
    return jsonify({"image": base64.b64encode(buffer).decode()})


@app.route("/static/<path:path>")
def send_static(path):
    return send_from_directory("static", path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
