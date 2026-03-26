import os
import tempfile
import subprocess
import shutil
from datetime import datetime

import streamlit as st
import cv2
import imageio_ffmpeg
from ultralytics import YOLO


# =========================
# Page UI
# =========================
st.set_page_config(page_title="Drone Detection", layout="centered")
st.title("🛸 Drone & Bird Detection")
st.write("ارفع فيديو، والنظام يحلل الفيديو ويطلع لك النتيجة بشكل أسرع.")

st.markdown("""
<style>
video {
    max-width: 260px !important;
    width: 100% !important;
    height: auto !important;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)


# =========================
# Model
# =========================
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
ALLOWED = {"drone", "bird"}


# =========================
# Sidebar Controls
# =========================
st.sidebar.header("⚙️ Settings")

conf_thres = st.sidebar.slider("Confidence", 0.05, 0.95, 0.30, 0.05)
iou_thres = st.sidebar.slider("IoU", 0.05, 0.95, 0.50, 0.05)
imgsz = st.sidebar.selectbox("Image Size", [320, 416, 512, 640], index=1)
skip_rate = st.sidebar.selectbox("Process every Nth frame", [1, 2, 3, 4, 5], index=2)
show_preview = st.sidebar.checkbox("Show live preview while processing", value=False)
save_drone_snapshots = st.sidebar.checkbox("Save drone snapshots", value=True)

uploaded = st.file_uploader("📤 ارفع فيديو", type=["mp4", "mov", "avi", "mkv"])

if uploaded is None:
    st.info("ارفع فيديو عشان نبدأ.")
    st.stop()


# =========================
# Save input
# =========================
tmp_dir = tempfile.mkdtemp()
input_path = os.path.join(tmp_dir, uploaded.name)
snapshots_dir = os.path.join(tmp_dir, "drone_snapshots")
os.makedirs(snapshots_dir, exist_ok=True)

with open(input_path, "wb") as f:
    f.write(uploaded.read())

st.success("✅ تم رفع الفيديو")


# =========================
# Show input video
# =========================
st.subheader("🎬 الفيديو الأصلي")
st.video(input_path)
st.divider()


# =========================
# Read video
# =========================
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    st.error("❌ ما قدرت أفتح الفيديو.")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    st.stop()

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 25

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

raw_output_path = os.path.join(tmp_dir, "output_raw.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))


# =========================
# Status UI
# =========================
st.subheader("⚙️ جاري المعالجة...")
progress = st.progress(0)
status = st.empty()
stats_box = st.empty()
preview_box = st.empty()

frame_idx = 0
processed_frames = 0
drone_count = 0
bird_count = 0
snapshot_count = 0


# =========================
# Processing
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # Skip frames for speed
    if skip_rate > 1 and (frame_idx % skip_rate != 0):
        writer.write(frame)
        if total_frames > 0:
            p = min(frame_idx / total_frames, 1.0)
            progress.progress(int(p * 100))
        continue

    processed_frames += 1
    drone_found_in_frame = False

    try:
        results = model.predict(
            frame,
            conf=conf_thres,
            iou=iou_thres,
            imgsz=imgsz,
            verbose=False
        )
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        break

    if results and len(results) > 0:
        r = results[0]

        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                cls = int(b.cls[0]) if b.cls is not None else -1
                name = names.get(cls, str(cls)).lower()

                if name not in ALLOWED:
                    continue

                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0]) if b.conf is not None else 0.0

                if name == "drone":
                    color = (0, 0, 255)
                    drone_count += 1
                    drone_found_in_frame = True
                else:
                    color = (0, 255, 0)
                    bird_count += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                txt = f"{name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                y_top = max(y1 - th - 10, 0)

                cv2.rectangle(frame, (x1, y_top), (x1 + tw + 8, y1), color, -1)
                cv2.putText(
                    frame,
                    txt,
                    (x1 + 4, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA
                )

    # Save snapshot if drone detected
    if save_drone_snapshots and drone_found_in_frame:
        snapshot_name = f"drone_{frame_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        snapshot_path = os.path.join(snapshots_dir, snapshot_name)
        cv2.imwrite(snapshot_path, frame)
        snapshot_count += 1

    writer.write(frame)

    if total_frames > 0:
        p = min(frame_idx / total_frames, 1.0)
        progress.progress(int(p * 100))
        status.write(f"Processing frame {frame_idx}/{total_frames} ...")
    else:
        status.write(f"Processing frame {frame_idx} ...")

    stats_box.info(
        f"""
**Stats**
- Processed frames: {processed_frames}
- Drone detections: {drone_count}
- Bird detections: {bird_count}
- Saved drone snapshots: {snapshot_count}
        """
    )

    if show_preview:
        preview_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preview_box.image(preview_rgb, caption="Live preview", use_container_width=True)


cap.release()
writer.release()

progress.progress(100)
status.success("✅ Finished!")


# =========================
# Convert to H264
# =========================
final_output_path = os.path.join(tmp_dir, "output_h264.mp4")

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

try:
    subprocess.run([
        ffmpeg, "-y",
        "-i", raw_output_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        final_output_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
except Exception as e:
    st.warning(f"⚠️ FFmpeg conversion failed, using raw output instead. Details: {e}")
    final_output_path = raw_output_path


# =========================
# Show output
# =========================
st.subheader("📌 الفيديو الناتج")

with open(final_output_path, "rb") as f:
    out_bytes = f.read()

st.video(out_bytes)

st.download_button(
    "⬇️ Download result video",
    data=out_bytes,
    file_name="drone_detection_output.mp4",
    mime="video/mp4"
)


# =========================
# Download snapshots as ZIP
# =========================
if save_drone_snapshots and os.path.exists(snapshots_dir):
    snapshot_files = os.listdir(snapshots_dir)
    if snapshot_files:
        zip_base = os.path.join(tmp_dir, "drone_snapshots")
        zip_path = shutil.make_archive(zip_base, 'zip', snapshots_dir)

        with open(zip_path, "rb") as zf:
            st.download_button(
                "📦 Download drone snapshots (ZIP)",
                data=zf.read(),
                file_name="drone_snapshots.zip",
                mime="application/zip"
            )

st.success("✅ انتهت المعالجة بنجاح")
