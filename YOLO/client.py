import cv2
import asyncio
import websockets
import base64
import numpy as np

SERVER_URL = "ws://192.168.68.201:5000/ws"


async def send_video():
    async with websockets.connect(SERVER_URL) as websocket:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            _, buffer = cv2.imencode(".jpg", frame)
            encoded_frame = base64.b64encode(buffer).decode("utf-8")

            await websocket.send(encoded_frame)
            annotated_frame_data = await websocket.recv()

            annotated_frame = np.frombuffer(
                base64.b64decode(annotated_frame_data), np.uint8
            )
            annotated_frame = cv2.imdecode(annotated_frame, cv2.IMREAD_COLOR)

            cv2.imshow("Annotated Video", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(send_video())
