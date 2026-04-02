import tempfile
from pathlib import Path

import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO

st.set_page_config(page_title="Drone & Bird Detection", layout="wide")
st.title("🛸 Drone & Bird Detection")

# =========================
# Load model once
# =========================
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# =========================
# Sidebar settings
# =========================
st.sidebar.header("Settings")
conf = st.sidebar.slider("Confidence", 0.05, 0.95, 0.30, 0.05)
imgsz = st.sidebar.selectbox("Image Size", [320, 416, 512, 640], index=1)

mode = st.radio(
    "Choose input mode:",
    ["Live Camera", "Upload Video"],
    horizontal=True
)

# =========================
# Helper function
# =========================
def detect_and_annotate(frame):
    results = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
    annotated = results[0].plot()
    return annotated

# =========================
# Live Camera Mode
# =========================
if mode == "Live Camera":
    st.subheader("📷 Live Camera")

    class VideoProcessor(VideoTransformerBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            annotated = detect_and_annotate(img)
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="drone-live",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# =========================
# Upload Video Mode
# =========================
else:
    st.subheader("🎥 Upload Video")

    uploaded_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded_file is None:
        st.info("Upload a video to start.")
        st.stop()

    # Save uploaded video temporarily
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(uploaded_file.read())
    temp_input_path = temp_input.name
    temp_input.close()

    st.video(temp_input_path)

    if st.button("Start Detection"):
        cap = cv2.VideoCapture(temp_input_path)

        if not cap.isOpened():
            st.error("Could not open uploaded video.")
            st.stop()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25

        temp_output_path = str(Path(temp_input_path).with_name("output_detected.mp4"))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        preview = st.empty()
        progress_text = st.empty()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            annotated = detect_and_annotate(frame)
            out.write(annotated)

            preview.image(annotated, channels="BGR", caption="Processing...")
            progress_text.write(f"Processing frame {frame_num}/{total_frames}")

        cap.release()
        out.release()

        st.success("Detection finished.")
        st.video(temp_output_path)

        with open(temp_output_path, "rb") as f:
            st.download_button(
                "Download Result",
                data=f,
                file_name="detected_output.mp4",
                mime="video/mp4"
            )
