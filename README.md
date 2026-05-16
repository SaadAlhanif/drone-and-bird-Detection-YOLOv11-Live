#  Drone and Bird Detection (YOLOv11) — Streamlit App

This project is a simple **Streamlit web app** that lets the user **upload a video** and runs **drone detection** using a trained **YOLOv11 (Ultralytics)** model.  
The app outputs a new video with **bounding boxes + “Drone” label** drawn on detected objects.

---

## ✅ Features
- Upload video (`.mp4`, `.mov`, `.avi`, `.mkv`)
- Run YOLO detection frame-by-frame
- Draw bounding box + label **Drone** + confidence score
- Preview the processed video inside Streamlit
- Download the output video

---

## 📁 Project Structure
```text
your-repo/
├── app.py
├── best.pt
├── requirements.txt
└── README.md
