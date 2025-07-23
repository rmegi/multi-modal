# 🧠 YOLO-FastAPI

A FastAPI application that performs real-time object detection using [Ultralytics YOLO](https://docs.ultralytics.com/) models. Video frames are streamed via GStreamer and processed using YOLO inference in Python.

---

## 🚀 Features

- ✅ FastAPI backend with async inference loop
- 🎥 Real-time video input via GStreamer (UDP or RTSP)
- 🧠 YOLOv8/YOLO-NAS object detection support (Ultralytics)
- 🐳 Docker + Docker Compose ready
- 🔌 Host networking for low-latency access to GStreamer pipelines

---

## 📡 GStreamer Pipeline

Make sure your GStreamer pipeline is sending frames the server IP:
```bash
curl -X POST http://192.168.68.124:5000/set_rcu_ip

