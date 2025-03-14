import cv2
import numpy as np
import os
from ultralytics import YOLO
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict

# Replace with your RTSP URL
rtsp_url = "rtsp://192.168.100.10/live/1"

# Open a connection to the RTSP stream
cap = cv2.VideoCapture(rtsp_url)


track_history = defaultdict(lambda: [])
model = YOLO("yolov10s.pt")
names = model.model.names

if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
else:
    print("RTSP stream opened successfully.")

# Read the first frame to initialize background
ret, frame1 = cap.read()

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

if not ret:
    print("Failed to retrieve frame. Exiting...")
    cap.release()
    cv2.destroyAllWindows()
    exit()


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to retrieve frame. Exiting...")
        break

    if success:
        results = model.track(frame, persist=True, verbose=False)
        boxes = results[0].boxes.xyxy.cpu()

        if results[0].boxes.id is not None:
            # Extract prediction results
            clss = results[0].boxes.cls.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.float().cpu().tolist()
             
            # Annotator Init
            annotator = Annotator(frame, line_width=2)
            
            for box, cls, track_id in zip(boxes, clss, track_ids):
                annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

                # Store tracking history
                track = track_history[track_id]
                track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                if len(track) > 30:
                    track.pop(0)
                
                # Plot tracks
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)
                cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)


    # Display the frames
    cv2.imshow("Original Frame", frame)
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break


# Release resources
cap.release()
cv2.destroyAllWindows()
