import cv2
from ultralytics import YOLO
from collections import Counter

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # YOLOv8 nano model

# Open webcam
cap = cv2.VideoCapture(0)  # Change to video file path if needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Extract detected objects
    class_counts = Counter(results[0].boxes.cls.tolist())  # Count occurrences of each class
    
    # Get class names
    class_names = model.names  # Get YOLO class labels

    # Create text with object count
    count_text = ", ".join(f"{int(count)} {class_names[int(cls)]}(s)" for cls, count in class_counts.items())

    # Plot results on frame
    annotated_frame = results[0].plot()

    # Display object count
    cv2.putText(annotated_frame, count_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the output
    cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
