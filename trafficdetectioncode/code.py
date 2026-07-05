import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Vehicle classes to detect
vehicle_classes = ['car', 'truck', 'bus', 'motorbike', 'bicycle']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    count = 0

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in vehicle_classes:
                count += 1

                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Put label 
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

    # Traffic density logic
    if count < 10:
        density = "Low Traffic"
    elif count < 20:
        density = "Medium Traffic"
    else:
        density = "High Traffic"

    # Display vehicle count
    cv2.putText(frame, f"Vehicle Count: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2)

    # Display density
    cv2.putText(frame, f"Density: {density}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 0), 2)

    # Show output
    cv2.imshow("Traffic Detection (Webcam)", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()