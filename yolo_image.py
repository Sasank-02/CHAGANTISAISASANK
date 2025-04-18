from ultralytics import YOLO
import cv2
import os

# Path to input image
image_path = r"I:\MTech\Deep Learning\Project\YOLO\Object dection using image\images\baggage_claim.jpg"


# Load the YOLOv8 model (you can use yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
print("[INFO] Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")  # or 'yolov8s.pt', 'yolov8m.pt', etc.

# Run inference on the image
print("[INFO] Running inference...")
results = model(image_path, conf=0.5)

# Plot the results on the image
print("[INFO] Drawing and saving results...")
annotated_frame = results[0].plot()

# Output path and save
output_dir = r"I:\MTech\Deep Learning\Project\YOLO\Object dection using image\Output"
os.makedirs(output_dir, exist_ok=True)

# Save with same name
filename = os.path.basename(image_path)
output_path = os.path.join(output_dir, filename)
cv2.imwrite(output_path, annotated_frame)
print(f"[INFO] Output image saved to: {output_path}")

# Show the image
cv2.imshow("YOLOv8 Detection", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
