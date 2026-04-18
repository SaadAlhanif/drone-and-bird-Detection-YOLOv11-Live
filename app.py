import os
import cv2
import time
import base64
import sqlite3
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

# 🔥 رابط الموديل (حط ID حقك)
MODEL_URL = "https://drive.google.com/file/d/1Bd0EvtNsagapzoDQ1zMPKePceyjlJ6oJ/view?usp=drive_link"
if not os.path.exists("best.pt"):
    print("Downloading model...")
    gdown.download(MODEL_URL, "best.pt", quiet=False)

model = YOLO("best.pt")

last_saved_time_live = 0


# 🧠 Database
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


# 🎯 رسم البوكس
def draw_boxes(frame, results):
    annotated = frame.copy()
    detections = []

    if results[0].boxes is None:
        return annotated, detections

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls_id].lower()

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if name == "drone":
            color = (0, 0, 255)  # 🔴
        elif name == "bird":
            color = (0, 255, 0)  # 🟢
        else:
            continue

        label = f"{name} {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        detections.append((name, conf))

    return annotated, detections


init_db()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/logs")
def logs():
    data = get_logs()
    return render_template("logs.html", detections=data)


# 📹 رفع فيديو
@app.route("/upload_video", methods=["POST"])
def upload_video():
    file = request.files["video"]

    filename = str(int(time.time())) + "_" + file.filename
    input_path = os.path.join(UPLOAD_FOLDER, filename)

    # 🔥 استخدم AVI (يحل مشكلة العرض)
    output_filename = "out_" + filename + ".avi"
    output_path = os.path.join(RESULT_FOLDER, output_filename)

    file.save(input_path)

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        return "Video error"

    width = int(cap.get(3))
    height = int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, 25, (width, height))

    last_saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.3, imgsz=320)
        annotated, detections = draw_boxes(frame, results)

        for name, conf in detections:
            if name == "drone":
                if time.time() - last_saved > 5:
                    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    img_path = os.path.join(SNAPSHOT_FOLDER, f"drone_{ts}.jpg")

                    cv2.imwrite(img_path, annotated)

                    insert_detection(
                        "drone",
                        conf,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        img_path,
                        "upload"
                    )

                    last_saved = time.time()
                break

        out.write(annotated)

    cap.release()
    out.release()

    return render_template(
        "index.html",
        uploaded_video="/" + input_path,
        result_video="/" + output_path
    )


# 📡 لايف
@app.route("/process_frame", methods=["POST"])
def process_frame():
    global last_saved_time_live

    data = request.json["image"]
    img = base64.b64decode(data.split(",")[1])
    np_img = np.frombuffer(img, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    results = model(frame, conf=0.3, imgsz=320)
    annotated, detections = draw_boxes(frame, results)

    for name, conf in detections:
        if name == "drone":
            if time.time() - last_saved_time_live > 5:
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                img_path = os.path.join(SNAPSHOT_FOLDER, f"live_{ts}.jpg")

                cv2.imwrite(img_path, annotated)

                insert_detection(
                    "drone",
                    conf,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    img_path,
                    "live"
                )

                last_saved_time_live = time.time()
            break

    _, buffer = cv2.imencode(".jpg", annotated)
    return jsonify({"image": base64.b64encode(buffer).decode()})


@app.route("/static/<path:path>")
def send_static(path):
    return send_from_directory("static", path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
