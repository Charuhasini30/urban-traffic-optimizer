import cv2
from ultralytics import YOLO
import time
from collections import defaultdict


model = YOLO("yolov8n.pt")  


video_path = "traffic_sample.mp4"  
cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))



frame_count = 0
vehicle_count = defaultdict(int)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model(frame, stream=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            if label in ['car', 'truck', 'bus', 'bike']:  # Adjust as needed
                vehicle_count[label] += 1
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Vehicle Detection", frame)
 

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()

print("Vehicle counts:", dict(vehicle_count))
print(" Give green signal to: West")
