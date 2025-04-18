import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load the most accurate YOLOv8 model
model = YOLO("yolov8x.pt")

# Initialize Deep SORT
tracker = DeepSort(max_age=30)

# Open webcam
cap = cv2.VideoCapture(0)

# Track unique person IDs
unique_person_ids = set()

# Dictionary to count total objects detected (by label)
object_counts = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame, conf=0.5)[0]

    detections = []
    current_frame_objects = []

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        label = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        current_frame_objects.append(label)

        # Update total object counts
        if label in object_counts:
            object_counts[label] += 1
        else:
            object_counts[label] = 1

        # For persons, add to Deep SORT for tracking
        if label == 'person':
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))
        else:
            # Draw other object boxes directly
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 150, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)

    # Track persons only via Deep SORT
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        x1, y1, x2, y2 = map(int, [l, t, r, b])

        unique_person_ids.add(track_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Person ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ==== Display Overlay ====
    overlay_x, overlay_y = 10, 30
    line_height = 25

    # Background rectangle
    overlay_height = (len(object_counts) + 2) * line_height + 10
    cv2.rectangle(frame, (5, 5), (300, 5 + overlay_height), (0, 0, 0), -1)

    # Text: Unique person count
    cv2.putText(frame, f"Unique Persons: {len(unique_person_ids)}",
                (overlay_x, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    overlay_y += line_height

    # Text: Other object counts
    for obj, count in object_counts.items():
        if obj != 'person':  # Skip person (already displayed above)
            cv2.putText(frame, f"{obj.capitalize()}: {count}",
                        (overlay_x, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
            overlay_y += line_height

    # Show final output
    cv2.imshow("YOLOv8 + Deep SORT: Person Tracking & Object Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
