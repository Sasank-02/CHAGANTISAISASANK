from ultralytics import YOLO
import cv2
import os

# Input and output paths
input_path = r"I:\MTech\Deep Learning\Project\YOLO\Object detection using video\videos\Footage Rainy Street.mp4"
output_dir = r"I:\MTech\Deep Learning\Project\YOLO\Object detection using video\output"
output_filename = os.path.basename(input_path)
output_path = os.path.join(output_dir, output_filename)

# Load YOLOv8 model (automatically downloads if not available)
print("[INFO] Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")  # You can change to yolov8s.pt, yolov8m.pt, etc.

# Run inference on video
print("[INFO] Running object detection on video...")
results = model.track(source=input_path, show=False, save=True, project=output_dir, name=os.path.splitext(output_filename)[0], tracker="bytetrack.yaml")

# Output path of saved video
saved_dir = os.path.join(output_dir, os.path.splitext(output_filename)[0])
print(f"[INFO] Output saved to: {saved_dir}")
