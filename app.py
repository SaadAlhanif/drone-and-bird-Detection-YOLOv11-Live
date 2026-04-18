import os
import cv2
import time
import base64
import sqlite3
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
SNAPSHOT_FOLDER = "static/snapshots"
DB_NAME = "detections.db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

model = YOLO("best.pt")

# يمنع تكرار حفظ نفس الدرون كل فريم
last_saved_time_live = 0


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        object_type TEXT NOT NULL,
        confidence REAL NOT NULL,
        detected_at TEXT NOT NULL,
        image_path TEXT NOT NULL,
        source TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()


def insert_detection(object_type, confidence, detected_at, image_path, source):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO detections (object_type, confidence, detected_at, image_path, source)
    VALUES (?, ?, ?, ?, ?)
    """, (object_type, confidence, detected_at, image_path, source))
    conn.commit()
    conn.close()


def get_all_detections():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, object_type, confidence, detected_at, image_path, source
    FROM detections
    ORDER BY id DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows


def draw_custom_boxes(frame, results):
    """
    Drone -> red
    Bird  -> green
    """
    annotated = frame.copy()

    if results[0].boxes is None:
        return annotated, []

    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = str(model.names[cls_id]).lower()

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if class_name == "drone":
            color = (0, 0, 255)  # red in BGR
        elif class_name == "bird":
            color = (0, 255, 0)  # green in BGR
        else:
            color = (255, 255, 0)

        label = f"{class_name} {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        detections.append({
            "class_name": class_name,
            "confidence": conf
        })

    return annotated, detections


init_db()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/logs")
def logs():
    rows = get_all_detections()
    return render_template("logs.html", detections=rows)


@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return "No video uploaded", 400

    file = request.files["video"]
    if file.filename == "":
        return "No selected video", 400

    filename = f"{int(time.time())}_{file.filename}"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_filename = f"result_{filename}"
    output_path = os.path.join(RESULT_FOLDER, output_filename)

    file.save(input_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return "Could not open uploaded video", 400

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    last_saved_time_upload = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.3, imgsz=416, verbose=False)
        annotated, detections = draw_custom_boxes(frame, results)

        # نسجل فقط drone
        for det in detections:
            if det["class_name"] == "drone":
                current_time = time.time()

                if current_time - last_saved_time_upload > 5:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    image_name = f"upload_drone_{timestamp}.jpg"
                    image_path = os.path.join(SNAPSHOT_FOLDER, image_name)

                    cv2.imwrite(image_path, annotated)

                    insert_detection(
                        object_type="drone",
                        confidence=det["confidence"],
                        detected_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        image_path=image_path,
                        source="upload"
                    )

                    last_saved_time_upload = current_time
                break

        out.write(annotated)

    cap.release()
    out.release()

    return render_template(
        "index.html",
        uploaded_video="/" + input_path,
        result_video="/" + output_path
    )


@app.route("/process_frame", methods=["POST"])
def process_frame():
    global last_saved_time_live

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image received"}), 400

    image_data = data["image"]
    if "," in image_data:
        image_data = image_data.split(",")[1]

    image_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid frame"}), 400

    results = model(frame, conf=0.3, imgsz=416, verbose=False)
    annotated, detections = draw_custom_boxes(frame, results)

    # نسجل فقط drone
    for det in detections:
        if det["class_name"] == "drone":
            current_time = time.time()

            if current_time - last_saved_time_live > 5:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                image_name = f"live_drone_{timestamp}.jpg"
                image_path = os.path.join(SNAPSHOT_FOLDER, image_name)

                cv2.imwrite(image_path, annotated)

                insert_detection(
                    object_type="drone",
                    confidence=det["confidence"],
                    detected_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    image_path=image_path,
                    source="live"
                )

                last_saved_time_live = current_time
            break

    _, buffer = cv2.imencode(".jpg", annotated)
    encoded_image = base64.b64encode(buffer).decode("utf-8")

    return jsonify({"image": encoded_image})


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
