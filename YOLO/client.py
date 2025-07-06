import cv2
import asyncio
import websockets
import base64
import numpy as np
import json # Import json to handle structured data from the server
import time # Import time for FPS control

# Define the server URL. Make sure this matches your server's address and port.
SERVER_URL = "ws://192.168.68.201:5000/ws"

async def send_video():
    """
    Connects to the WebSocket server, captures video from the webcam,
    sends frames for object detection every N frames, receives bounding box data,
    and displays the annotated video stream at a higher FPS.
    """
    print(f"Attempting to connect to WebSocket server at {SERVER_URL}...")
    try:
        async with websockets.connect(SERVER_URL, ping_interval=None) as websocket:
            print("Connected to WebSocket server.")
            cap = cv2.VideoCapture(0) # Open the default webcam

            if not cap.isOpened():
                print("Error: Could not open webcam.")
                return

            # --- Configure Display and Detection FPS ---
            TARGET_DISPLAY_FPS = 30 # Desired frames per second for display
            TARGET_DETECTION_FPS = 10 # Desired frames per second for sending to server
            
            # Calculate the frame skip ratio for detection
            # For example, if DISPLAY=30, DETECTION=10, then skip_frames = 30/10 = 3
            # This means we send every 3rd frame for detection.
            if TARGET_DETECTION_FPS <= 0 or TARGET_DISPLAY_FPS < TARGET_DETECTION_FPS:
                print("Error: Invalid FPS configuration. TARGET_DETECTION_FPS must be positive and less than or equal to TARGET_DISPLAY_FPS.")
                return
            
            # Ensure skip_frames is an integer and at least 1
            skip_frames = max(1, int(TARGET_DISPLAY_FPS / TARGET_DETECTION_FPS))
            
            # Calculate the time delay needed per frame for display to achieve the target display FPS
            FRAME_DISPLAY_DELAY = 1.0 / TARGET_DISPLAY_FPS

            print(f"Displaying video at approximately {TARGET_DISPLAY_FPS} FPS.")
            print(f"Sending frames to server for detection every {skip_frames} frames (approx. {TARGET_DETECTION_FPS} FPS detection).")
            print("Press 'q' to quit.")

            frame_count = 0
            last_bounding_boxes = [] # Store the last received bounding boxes

            while True:
                frame_start_time = time.time() # Record the start time of the current frame processing

                ret, frame = cap.read() # Read a frame from the webcam
                if not ret:
                    print("Error: Could not read frame from webcam. Exiting.")
                    break

                display_frame = frame.copy() # Create a copy for drawing annotations

                # --- Conditional Server Call for Detection ---
                if frame_count % skip_frames == 0:
                    # Encode the frame to JPEG format for efficient transmission
                    _, buffer = cv2.imencode('.jpg', frame)
                    # Convert the buffer to a base64 string
                    encoded_frame = base64.b64encode(buffer).decode("utf-8")

                    try:
                        # Send the encoded frame to the server
                        await websocket.send(encoded_frame)

                        # Receive the bounding box data as a JSON string from the server
                        bbox_data_json = await websocket.recv()
                        # Parse the JSON string into a Python list of dictionaries
                        last_bounding_boxes = json.loads(bbox_data_json)
                        
                    except websockets.exceptions.ConnectionClosedOK:
                        print("Server closed the connection gracefully.")
                        break
                    except websockets.exceptions.ConnectionClosedError as e:
                        print(f"Server closed the connection with an error: {e}")
                        break
                    except json.JSONDecodeError:
                        print("Error: Received invalid JSON data from server. Using last known bounding boxes.")
                        # If JSON decoding fails, we continue using the last_bounding_boxes
                        pass
                    except Exception as e:
                        print(f"Error during server communication: {e}. Using last known bounding boxes.")
                        pass # Continue with last known bounding boxes if an error occurs

                # --- Draw Annotations using last_bounding_boxes ---
                for bbox_info in last_bounding_boxes:
                    class_name = bbox_info.get("class_name", "Unknown")
                    confidence = bbox_info.get("confidence", 0.0)
                    box = bbox_info.get("box", {})
                    x1 = int(box.get("x1", 0))
                    y1 = int(box.get("y1", 0))
                    x2 = int(box.get("x2", 0))
                    y2 = int(box.get("y2", 0))

                    # Draw the rectangle (bounding box) on the frame
                    # Color is green (0, 255, 0), thickness is 2
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Put the class name and confidence text above the bounding box
                    # Font, scale, color, and thickness are set for readability
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(display_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Display the frame with annotations
                cv2.imshow("Live Video with Object Detections", display_frame)

                # Increment frame count
                frame_count += 1

                # Check for 'q' key press to quit the application
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("'q' pressed. Exiting.")
                    break

                # --- FPS Control for Display ---
                frame_end_time = time.time() # Record the end time of the current frame processing
                # Calculate the time taken for this frame
                time_taken = frame_end_time - frame_start_time
                # Calculate the remaining time to sleep to meet the target display FPS
                sleep_time = FRAME_DISPLAY_DELAY - time_taken

                if sleep_time > 0:
                    await asyncio.sleep(sleep_time) # Pause for the remaining time
                # else:
                #     print(f"Warning: Running below target display FPS. Frame processing took {time_taken:.4f}s")

    except ConnectionRefusedError:
        print(f"Error: Connection refused. Is the server running at {SERVER_URL}?")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Release the webcam and destroy all OpenCV windows
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("Webcam released and windows closed.")

if __name__ == "__main__":
    # Run the asynchronous function
    asyncio.run(send_video())
