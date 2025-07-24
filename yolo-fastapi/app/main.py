import asyncio
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread, Lock
from queue import Queue, Empty
import time
import gi
import logging
import sys
from ultralytics import YOLOE
import torch
from prompt_manager import PromptManager

# Initialize GStreamer
gi.require_version("Gst", "1.0")
from gi.repository import Gst

Gst.init(None)
print(torch.cuda.is_available())

prompt_path = "prompt.json"
prompt_manager = PromptManager(prompt_path)

model = YOLOE("yoloe-11s-seg.pt")
OBJECT_CLASSES = prompt_manager.get_prompts()
model.set_classes(OBJECT_CLASSES, model.get_text_pe(OBJECT_CLASSES))


# Async inference
async def detect_and_annotate(frame: np.ndarray) -> np.ndarray:
    results = model.predict(frame, verbose=False)
    for r in results:
        r.masks = None  # Skip segmentation
    return results[0].plot()


def await_detect_and_annotate_sync(frame: np.ndarray) -> np.ndarray:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    annotated_frame = loop.run_until_complete(detect_and_annotate(frame))
    loop.close()
    return annotated_frame


# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler(sys.stdout)],
)
app_logger = logging.getLogger("FastAPI")
gstreamer_logger = logging.getLogger("GStreamer")

# GStreamer log redirect


def gstreamer_log_handler(level, domain, message):
    log_level = {
        Gst.DebugLevel.ERROR: logging.ERROR,
        Gst.DebugLevel.WARNING: logging.WARNING,
        Gst.DebugLevel.INFO: logging.INFO,
        Gst.DebugLevel.DEBUG: logging.DEBUG,
        Gst.DebugLevel.LOG: logging.DEBUG,
        Gst.DebugLevel.TRACE: logging.DEBUG,
    }.get(level, logging.INFO)
    gstreamer_logger.log(log_level, f"{domain}: {message}")


Gst.debug_add_log_function(gstreamer_log_handler, None)
Gst.debug_set_default_threshold(3)

# Shared data
frame_buffer = None
frame_lock = Lock()
latest_timestamp = 0
frame_queue = Queue(maxsize=10)
annotated_queue = Queue(maxsize=5)

# FPS tracking
frame_counter = 0
fps_timestamp = time.time()

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GStreamer pipeline


def create_gstreamer_pipeline(port):
    pipeline_str = f"""
        udpsrc port={port} !
        application/x-rtp, payload=96 !
        rtph264depay !
        avdec_h264 !
        videoconvert ! video/x-raw, format=BGR !
        appsink name=appsink0 sync=false
    """
    app_logger.info(f"Created GStreamer pipeline for port {port}")
    return Gst.parse_launch(pipeline_str)


def extract_frame(sample):
    buffer = sample.get_buffer()
    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        gstreamer_logger.error("Failed to map buffer.")
        return None
    try:
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_int("width")[1]
        height = structure.get_int("height")[1]
        frame = np.frombuffer(map_info.data, np.uint8).reshape((height, width, 3))
    except Exception as e:
        gstreamer_logger.error(f"Error extracting frame: {e}")
        frame = None
    finally:
        buffer.unmap(map_info)
    return frame


def inference_worker():
    while True:
        try:
            frame = frame_queue.get(timeout=1)
            annotated = await_detect_and_annotate_sync(frame)
            annotated_queue.put(annotated)
        except Empty:
            continue
        except Exception as e:
            app_logger.error(f"Inference error: {e}")


def generate_frames(pipeline):
    global frame_buffer, latest_timestamp, frame_counter, fps_timestamp
    frame_count = 0
    detection_interval = 1

    appsink = pipeline.get_by_name("appsink0")
    if not appsink:
        gstreamer_logger.error("Appsink element not found in the pipeline.")
        return

    print("🟢 Appsink is ready, starting to pull frames...")

    while True:
        sample = appsink.emit("pull-sample")
        if sample is None:
            gstreamer_logger.warning("⚠️ No sample pulled from appsink.")
            continue

        print("✅ Sample received from appsink.")

        frame = extract_frame(sample)
        if frame is not None:
            print(f"📸 Frame extracted: shape={frame.shape}")
        else:
            print("❌ Failed to extract frame.")
            continue

        frame_count += 1
        frame_counter += 1

        now = time.time()
        if now - fps_timestamp >= 1.0:
            print(f"📈 FPS: {frame_counter}")
            frame_counter = 0
            fps_timestamp = now

        if frame_count % detection_interval == 0:
            try:
                frame_queue.put_nowait(frame)
                print("🧠 Frame pushed to inference queue.")
            except:
                print("⚠️ Inference queue is full.")

        try:
            frame = annotated_queue.get_nowait()
            print("🟩 Retrieved annotated frame from queue.")
        except Empty:
            print("⏳ No annotated frame ready.")
            continue

        latest_timestamp = time.time()
        _, buffer = cv2.imencode(".jpg", frame)
        with frame_lock:
            frame_buffer = buffer.tobytes()


def start_pipeline():
    app_logger.info("🚀 Starting GStreamer pipeline...")
    port = 5004
    pipeline = create_gstreamer_pipeline(port)
    pipeline.set_state(Gst.State.PLAYING)
    print(f"🔌 Pipeline set to PLAYING on port {port}")
    Thread(target=generate_frames, args=(pipeline,), daemon=True).start()
    Thread(target=inference_worker, daemon=True).start()
    app_logger.info("✅ Pipeline and inference threads started.")


@app.on_event("startup")
def startup_event():
    start_pipeline()


@app.get("/update_prompt")
def update_prompt(new_classes: list[str]):
    prompt_manager.update_prompt(new_classes)
    return {"status": "success", "new_classes": prompt_manager.get_prompts()}


@app.get("/get_prompt")
def get_prompt():
    return {"status": "success", "classes": prompt_manager.get_prompts()}


@app.get("/video")
async def video_stream():
    def stream_frames():
        locked_frame = None
        while True:
            with frame_lock:
                if frame_buffer:
                    locked_frame = frame_buffer
            if locked_frame:
                current_time = time.time()
                if current_time - latest_timestamp <= 1.0:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + locked_frame + b"\r\n\r\n"
                    )
            time.sleep(0.03)

    return StreamingResponse(
        stream_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/video/timestamp")
async def get_timestamp():
    return {"timestamp": latest_timestamp}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/snapshot")
async def snapshot():
    with frame_lock:
        if frame_buffer:
            return Response(content=frame_buffer, media_type="image/jpeg")
        return Response(content=b"", media_type="image/jpeg")


if __name__ == "__main__":
    app_logger.info("Starting FastAPI application...")
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
