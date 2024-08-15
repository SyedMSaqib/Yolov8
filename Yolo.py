from ultralytics import YOLO
import cv2

# Load the model
model = YOLO("yolov8n.pt")  # Load a pretrained YOLOv8 model

# Load an image
image_path = "Untitled.jpeg"  # Replace with the path to your input image
image = cv2.imread(image_path)

# Perform inference
results = model(image_path)

# Filter results to keep only 'person' class detections (class ID for 'person' is 0 in COCO dataset)
people_detections = [det for det in results[0].boxes if int(det.cls) == 15]

# Display or process the filtered detections
if people_detections:
    print(f"Detected {len(people_detections)} people in the image.")
    
    # Draw bounding boxes for people
    for det in people_detections:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, det.xyxy[0])  # Get coordinates of the bounding box
        confidence = float(det.conf)  # Get confidence score
        
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display confidence score
        label = f"Person: {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with bounding boxes
    cv2.imshow("People Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No people detected in the image.")
