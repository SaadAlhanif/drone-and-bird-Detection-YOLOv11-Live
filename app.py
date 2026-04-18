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


if name == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
