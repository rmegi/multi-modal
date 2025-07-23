from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLOE
import numpy as np
import cv2

app = FastAPI()
model = YOLOE("yoloe-11s-seg.pt")
OBJECT_CLASSES = ["person", "weapon", "stairs", "house", "door", "window"]
model.set_classes(OBJECT_CLASSES, model.get_text_pe(OBJECT_CLASSES))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = model.predict(img)
    names = results[0].names
    detected = [names[int(cls)] for cls in results[0].boxes.cls]
    return {"labels": detected}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
