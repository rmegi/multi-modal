# Server for Object Detection
Uses [YOLOE](https://docs.ultralytics.com/models/yoloe/) to recives a video stream and return an annotated video stream with the desired objects to detect.

The list of objects is defined in the begining of the server and you can change it. Currently it's:

```OBJECT_CLASSES = ["person", "weapon", "stairs", "house", "door", "window"]```

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn server:app --host 0.0.0.0 --port 5000
```

This will start the server.

If this is the first time you are running it, it will try to download the models it needs, so it can take a few minutes.

## Testing
```bash
python client.py
``` 
This will run a client that shows the webcam with annotations using the server.

Make sure to run the server before.

# Do Not Forget
## You Only Live Once
YOLO