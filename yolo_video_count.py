import argparse
import cv2
import os
from ultralytics import YOLO  # Import YOLOv8 from ultralytics

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default="I:\\MTech\\Deep Learning\\Project\\YOLO\\Object detection using video\\videos\\Footage Rainy Street.mp4", help="Path to input video")
ap.add_argument("-o", "--output", default="I:\\MTech\\Deep Learning\\Project\\YOLO\\Object detection using video\\output", help="Path to output video folder")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum confidence threshold")
args = vars(ap.parse_args())

# Load YOLOv8 model (use yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
model = YOLO("yolov8n.pt")  # You can change this to another YOLOv8 variant

# Open video file
vs = cv2.VideoCapture(args["input"])
writer = None

# Dictionary to count objects detected by type
object_counts = {}

while True:
    ret, frame = vs.read()
    if not ret:
        break  # End of video if no frame is captured
    
    # Run YOLOv8 object detection
    results = model(frame, conf=args["confidence"])
    
    # Draw bounding boxes and labels on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class ID
            label = model.names[cls]

            # Update object counts
            if label in object_counts:
                object_counts[label] += 1
            else:
                object_counts[label] = 1

            # Draw rectangle and label
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Initialize video writer if not already done
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        output_path = os.path.join(args["output"], "trailer_output.avi")
        writer = cv2.VideoWriter(output_path, fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)
    
    # Display object counts on the frame
    overlay_x, overlay_y = 10, 30
    line_height = 30

    # Create background for the overlay
    overlay_height = (len(object_counts) + 1) * line_height + 10
    cv2.rectangle(frame, (5, 5), (300, 5 + overlay_height), (0, 0, 0), -1)

    # Show the total objects and their types
    cv2.putText(frame, f"Total Objects: {sum(object_counts.values())}",
                (overlay_x, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    overlay_y += line_height

    for obj, count in object_counts.items():
        cv2.putText(frame, f"{obj.capitalize()}: {count}",
                    (overlay_x, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        overlay_y += line_height
    
    # Write the processed frame to output video
    writer.write(frame)

# Release resources
print("[INFO] Cleaning up...")
writer.release()
vs.release()
