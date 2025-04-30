import cv2
from ultralytics import YOLO
import numpy as np


model = YOLO('yolov8n.pt')


video_path = ("C:\lnctbus3.mp4")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


vehicle_classes = ["car", "motorbike", "bus", "truck"]


frame_rate = cap.get(cv2.CAP_PROP_FPS)
pixels_per_meter = 10


previous_positions = {}

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read the frame.")
        break


    results = model(frame)


    current_positions = {}
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = int(box.cls[0])
        class_name = model.names[cls]


        if class_name in vehicle_classes:

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            current_positions[(center_x, center_y)] = class_name


            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    for (current_x, current_y), class_name in current_positions.items():
        for (prev_x, prev_y), prev_class_name in previous_positions.items():
            if class_name == prev_class_name:

                distance_pixels = np.sqrt((current_x - prev_x) ** 2 + (current_y - prev_y) ** 2)

                distance_meters = distance_pixels / pixels_per_meter


                speed_mps = distance_meters * frame_rate
                speed_kph = speed_mps * 3.6


                cv2.putText(frame, f"Speed: {speed_kph:.2f} km/h", (int(current_x), int(current_y) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    previous_positions = current_positions


    cv2.imshow("Vehicle Detection and Speed Estimation", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
