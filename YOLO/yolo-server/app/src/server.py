from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLOE
import cv2
import numpy as np
import base64

app = FastAPI()

model = YOLOE("yoloe-11l-seg.pt")
OBJECT_CLASSES = ["person", "weapon", "stairs", "house", "door", "window"]
model.set_classes(OBJECT_CLASSES, model.get_text_pe(OBJECT_CLASSES))


async def detect_and_annotate(frame: np.ndarray) -> np.ndarray:
    results = model.predict(frame, verbose=False)
    annotated_frame = results[0].plot()
    return annotated_frame


@app.websocket("/ws")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            frame_data = base64.b64decode(data)
            np_data = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

            annotated_frame = await detect_and_annotate(frame)
            _, buffer = cv2.imencode(".jpg", annotated_frame)
            encoded = base64.b64encode(buffer).decode("utf-8")

            await websocket.send_text(encoded)
    except WebSocketDisconnect:
        print("Client disconnected")
