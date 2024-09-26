# Import necessary libraries
import cv2
from ultralytics import YOLO

# Function to initialize YOLOv8 model
def initialize_yolov8(model_path='yolov8n.pt'):
    """
    Load YOLOv8 model from ultralytics package.
    
    Args:
        model_path (str): Path to YOLOv8 model. Default is 'yolov8n.pt' (pre-trained).

    Returns:
        YOLO model
    """
    # Load YOLOv8 model (n for small or custom trained model)
    model = YOLO(model_path)
    return model

# Perception Layer: Object detection and feature extraction
def perception_layer_yolov8(image, model):
    """
    Use YOLOv8 model to perform object detection on input image.
    
    Args:
        image (np.array): The input image (BGR format from OpenCV).
        model (YOLO): Pre-trained YOLOv8 model.

    Returns:
        list: Detected objects with their bounding boxes, class labels, and confidence scores.
    """
    # Perform object detection using YOLOv8
    results = model(image)
    
    # Extract detection results
    detections = []
    for result in results:
        # Loop through each detected object
        for box in result.boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            # Get confidence score
            confidence = box.conf[0]
            # Get class label
            class_id = int(box.cls[0])
            # Append to list as a tuple (bounding box, confidence, class_id)
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(confidence),
                'class_id': class_id
            })
    return detections

# Load an image (for testing)
def load_image(image_path):
    """
    Load an image using OpenCV.
    
    Args:
        image_path (str): Path to the image file.

    Returns:
        np.array: Loaded image in BGR format.
    """
    image = cv2.imread(image_path)
    return image

# Draw bounding boxes on the image
def draw_detections(image, detections, class_names):
    """
    Draw bounding boxes and class labels on the detected objects.
    
    Args:
        image (np.array): The input image (BGR format).
        detections (list): List of detected objects with bounding boxes, class labels, and confidence scores.
        class_names (list): List of class names corresponding to class IDs.
    
    Returns:
        np.array: Image with bounding boxes and labels drawn.
    """
    for detection in detections:
        bbox = detection['bbox']
        class_id = detection['class_id']
        confidence = detection['confidence']
        
        # Draw bounding box
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # Add class label and confidence
        label = f"{class_names[class_id]}: {confidence:.2f}"
        cv2.putText(image, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

# Main Function: Run the Perception Layer
if __name__ == '__main__':
    # Initialize YOLOv8 model (you can also use custom-trained YOLOv8 models)
    model = initialize_yolov8()

    # Load an example image
    image_path = 'path/to/your/image.jpg'  # Replace with the actual image path
    image = load_image(image_path)

    # Perform object detection
    detections = perception_layer_yolov8(image, model)

    # Define class names (for COCO dataset if using a pre-trained YOLOv8 model)
    class_names = model.names

    # Draw the detections on the image
    image_with_detections = draw_detections(image, detections, class_names)

    # Display the image with detections
    cv2.imshow('Detected Objects', image_with_detections)
    cv2.waitKey(0)  # Press any key to close the window
    cv2.destroyAllWindows()

    # Save the image with detections
    output_image_path = 'output_with_detections.jpg'
    cv2.imwrite(output_image_path, image_with_detections)
    print(f"Output image saved at {output_image_path}")
