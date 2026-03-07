"""
Vehicle Counting & Classification Module
Uses YOLOv8 Nano for real-time vehicle detection and classification.
Supports: car, motorcycle, bus, truck
"""

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import base64

# COCO class IDs for vehicles
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

# Load model globally for performance
_model = None

def get_model():
    global _model
    if _model is None:
        _model = YOLO('yolov8l.pt')  # Upgraded to Large model
    return _model


def detect_vehicles(image_bytes):
    """
    Detect and classify vehicles in an image.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        dict with keys:
            - counts: dict of vehicle type -> count
            - detections: list of detection dicts (class, confidence, bbox)
            - annotated_image: base64 encoded annotated image
            - total: total vehicle count
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Could not decode image"}
    
    model = get_model()
    
    # Auto-detect device (GPU if available)
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Run inference with Large model and tuned parameters
    results = model(img, verbose=False, conf=0.45, iou=0.5, device=device)[0]
    
    counts = {v: 0 for v in VEHICLE_CLASSES.values()}
    detections = []
    annotated = img.copy()
    
    # Color map for vehicle types
    colors = {
        'car': (0, 255, 200),
        'motorcycle': (255, 100, 0),
        'bus': (0, 150, 255),
        'truck': (200, 0, 255)
    }
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in VEHICLE_CLASSES:
            vehicle_type = VEHICLE_CLASSES[cls_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            counts[vehicle_type] += 1
            detections.append({
                'class': vehicle_type,
                'confidence': round(confidence, 3),
                'bbox': [x1, y1, x2, y2]
            })
            
            # Draw bounding box
            color = colors.get(vehicle_type, (0, 255, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{vehicle_type} {confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Encode annotated image to base64
    _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    annotated_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        'counts': counts,
        'detections': detections,
        'annotated_image': annotated_b64,
        'total': sum(counts.values())
    }
