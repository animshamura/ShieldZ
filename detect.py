import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 pretrained model (can be custom trained on "suspicious activity" for better results)
model = YOLO("yolov8n.pt")  # You can train a custom model for 'thief', 'robber' classes

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Open webcam or video file
cap = cv2.VideoCapture(0)  # Use video path for recorded footage

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame)[0]
    detections = []

    for result in results.boxes:
        cls_id = int(result.cls)
        label = model.names[cls_id]
        conf = float(result.conf)
        x1, y1, x2, y2 = map(int, result.xyxy[0])

        # Filter only for person
        if label == "person" and conf > 0.5:
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

    # Track detections using DeepSORT
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Shielz Surveillance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
