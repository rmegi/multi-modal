from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLOE
import cv2
import numpy as np
import base64
from fastapi import UploadFile, File
from fastapi.responses import StreamingResponse
import io
import uvicorn

app = FastAPI()

model = YOLOE("yoloe-11l-seg.pt")
OBJECT_CLASSES = ["weapon", "stairs", "house", "door", "window"]
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

            # Run model and get results
            results = model.predict(frame, verbose=False)
            detections = results[0]

            # Print detected objects to console
            if detections.boxes is not None and len(detections.boxes) > 0:
                print("Detected objects:")
                for box in detections.boxes:
                    cls_id = int(box.cls[0])
                    label = OBJECT_CLASSES[cls_id]
                    conf = float(box.conf[0])
                    print(f"  - {label} ({conf:.2f})")
            else:
                print("No objects detected.")

            # Continue sending annotated image
            annotated_frame = detections.plot()
            _, buffer = cv2.imencode(".jpg", annotated_frame)
            encoded = base64.b64encode(buffer).decode("utf-8")
            await websocket.send_text(encoded)

    except WebSocketDisconnect:
        print("Client disconnected")


@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):
    # Read and decode the image
    image_bytes = await file.read()
    np_data = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    # Annotate with YOLOE
    annotated_frame = await detect_and_annotate(frame)

    # Encode annotated image to JPEG for response
    _, buffer = cv2.imencode(".jpg", annotated_frame)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")


@app.post("/detect-image-json")
async def detect_image(file: UploadFile = File(...)):
    # Read and decode the image
    image_bytes = await file.read()
    np_data = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    # Run YOLOE prediction
    results = model.predict(frame, verbose=False)
    detections = results[0]  # First image's results

    response_data = []
    for box in detections.boxes:
        cls_id = int(box.cls[0])
        label = OBJECT_CLASSES[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coords

        response_data.append(
            {"label": label, "confidence": conf, "bbox": [x1, y1, x2, y2]}
        )
    print(f"The data is {response_data}")
    return {"detections": response_data}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=9000, reload=True)
