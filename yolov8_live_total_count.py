import cv2
from ultralytics import YOLO
import random

# Load the most accurate YOLOv8 model (YOLOv8x)
model = YOLO("yolov8x.pt")  # Make sure this model file is available

# Open webcam (0 = default)
vs = cv2.VideoCapture(0)

# Store unique colors for each class
class_colors = {}

while True:
    ret, frame = vs.read()
    if not ret:
        break

    # Run detection with confidence threshold
    results = model(frame, conf=0.5)
    object_count = {}

    # Loop over results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Assign color if new label
            if label not in class_colors:
                class_colors[label] = (
                    random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)
                )

            # Update per-frame object count
            object_count[label] = object_count.get(label, 0) + 1

            # Draw bounding box and label
            color = class_colors[label]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Transparent info panel
    overlay = frame.copy()
    panel_width = 300
    panel_height = 40 + 30 * len(object_count)
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
    alpha = 0.5
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Display object count
    y = 30
    cv2.putText(frame, "Detected This Frame:", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 35

    total = 0
    for label, count in sorted(object_count.items()):
        total += count
        color = class_colors[label]
        cv2.putText(frame, f"{label}: {count}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        y += 30

    # Total count
    cv2.putText(frame, f"Total: {total}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Display the output
    cv2.imshow("YOLOv8x - Live Detection & Count", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
print("[INFO] Cleaning up...")
vs.release()
cv2.destroyAllWindows()
