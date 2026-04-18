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
    import subprocess


app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
SNAPSHOT_FOLDER = "static/snapshots"
DB_NAME = "detections.db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

# حط ID حق best.pt هنا
MODEL_URL = "https://drive.google.com/file/d/1Bd0EvtNsagapzoDQ1zMPKePceyjlJ6oJ/view?usp=drive_link"

if not os.path.exists("best.pt"):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, "best.pt", quiet=False)

model = YOLO("best.pt")

last_saved_time_live = 0


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
            color = (0, 0, 255)  # red
        elif name == "bird":
            color = (0, 255, 0)  # green
        else:
            continue

        label = f"{name} {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

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

    temp_output_path = os.path.join(
        RESULT_FOLDER,
        f"temp_{os.path.splitext(filename)[0]}.avi"
    )

    final_output_filename = f"out_{os.path.splitext(filename)[0]}.mp4"
    final_output_path = os.path.join(RESULT_FOLDER, final_output_filename)

    file.save(input_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return "Video error"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    last_saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.3, imgsz=320, verbose=False)
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


final_output_path = temp_output_path.replace(".mp4", "_final.mp4")

subprocess.call(
    f'ffmpeg -y -i "{temp_output_path}" -c:v libx264 -preset ultrafast -crf 23 -pix_fmt yuv420p "{final_output_path}"',
    shell=True
)

# استخدم الفيديو الجديد بدل القديم
output_video_path = final_output_path
    if os.path.exists(final_output_path):
        os.remove(final_output_path)

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", temp_output_path,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            final_output_path
        ],
        capture_output=True,
        text=True
    )

    print("FFMPEG RETURN CODE:", result.returncode)
    print("FFMPEG STDERR:", result.stderr)

    if result.returncode != 0:
        return f"FFmpeg conversion failed:\n{result.stderr}"

    return render_template(
        "index.html",
        uploaded_video=input_path.replace("static/", ""),
        result_video=final_output_path.replace("static/", "")
    )


@app.route("/process_frame", methods=["POST"])
def process_frame():
    global last_saved_time_live

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image received"}), 400

    image_data = data["image"]
    image_bytes = base64.b64decode(image_data.split(",")[1])
    np_img = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid frame"}), 400

    results = model(frame, conf=0.3, imgsz=320, verbose=False)
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
