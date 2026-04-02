import os
import cv2
import time
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO("best.pt")


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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.3, imgsz=416, verbose=False)
        annotated = results[0].plot()
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
    annotated = results[0].plot()

    _, buffer = cv2.imencode(".jpg", annotated)
    encoded_image = base64.b64encode(buffer).decode("utf-8")

    return jsonify({"image": encoded_image})


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
