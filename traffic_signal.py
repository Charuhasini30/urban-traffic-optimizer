from ultralytics import YOLO
import cv2
import numpy as np
import os


model = YOLO('yolov8n.pt')
image_path = 'intersection.jpg'
full_image = cv2.imread(image_path)
if full_image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")


os.makedirs('results', exist_ok=True)

height, width, _ = full_image.shape
mid_h, mid_w = height // 2, width // 2

regions = {
    'north': full_image[0:mid_h, 0:mid_w],
    'south': full_image[mid_h:, 0:mid_w],
    'east':  full_image[0:mid_h, mid_w:],
    'west':  full_image[mid_h:, mid_w:]
}

vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
vehicle_counts = {}

print("\n--- Detection Logs ---")


for direction, img in regions.items():
    results = model(img, verbose=False)[0]
    annotated = results.plot()

    
    output_path = f'results/{direction}_output.jpg'
    cv2.imwrite(output_path, annotated)


    count = 0
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        if label in vehicle_classes:
            count += 1
    
    vehicle_counts[direction] = count
    print(f"{direction.capitalize():<6}: {count} vehicles")

min_direction = min(vehicle_counts, key=vehicle_counts.get)

print("\n--- Vehicle Count per Direction ---")
for dir, cnt in vehicle_counts.items():
    print(f"{dir.capitalize():<6}: {cnt} vehicles")

print(f"\n Give GREEN signal to → {min_direction.upper()} (Least traffic)")
