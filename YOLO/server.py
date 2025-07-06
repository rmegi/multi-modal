from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLOE
import cv2
import numpy as np
import base64
import json # Import json for sending structured data
from typing import List, Dict

app = FastAPI()

# Load the YOLOE model once when the application starts
model = YOLOE("yoloe-11l-seg.pt")
OBJECT_CLASSES = ["person", "weapon", "stairs", "house", "door", "window"]
# Assuming model.set_classes expects class names and potentially a processed representation
# If model.get_text_pe() is for text embeddings, you might need to ensure its output is correct
# for the YOLOE model's specific set_classes method.
model.set_classes(OBJECT_CLASSES, model.get_text_pe(OBJECT_CLASSES))

# Function to perform detection and return bounding box data
async def detect_objects(frame: np.ndarray) -> List[Dict]:
    results = model.predict(frame, verbose=False)

    detections = []
    # Iterate through the results to extract bounding box, confidence, and class
    # The structure of results[0].boxes might vary slightly based on the Ultralytics version,
    # but generally, it provides access to xyxy (corners), confidences, and class IDs.
    if results and results[0].boxes:
        for box in results[0].boxes:
            # box.xyxy[0] gives [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = OBJECT_CLASSES[class_id] # Map class ID to class name

            detections.append({
                "class_name": class_name,
                "confidence": confidence,
                "box": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            })
    return detections

@app.websocket("/ws")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            frame_data = base64.b64decode(data)
            np_data = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

            # Perform detection and get bounding box data
            bounding_boxes = await detect_objects(frame)

            # Send the bounding box data as a JSON string
            await websocket.send_text(json.dumps(bounding_boxes))
    except WebSocketDisconnect:
        print("Client disconnected")