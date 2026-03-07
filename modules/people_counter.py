"""
People Counting & Gender Classification Module
Uses YOLOv8 Nano for person detection and counting.
Provides crowd density estimation and gender classification via DeepFace.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import base64

# COCO class ID for person
PERSON_CLASS_ID = 0

_model = None


def get_model():
    global _model
    if _model is None:
        _model = YOLO('yolov8l.pt')  # Upgraded to Large model
    return _model


def estimate_density(person_count, image_area):
    """Estimate crowd density level."""
    density = person_count / (image_area / 1000000)  # per megapixel
    if density < 2:
        return 'Low'
    elif density < 8:
        return 'Medium'
    elif density < 20:
        return 'High'
    else:
        return 'Very High'


def generate_heatmap(img, detections):
    """Generate a density heatmap overlay."""
    heatmap = np.zeros(img.shape[:2], dtype=np.float32)
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # Add gaussian-like heat at center of each person
        radius = max(30, (x2 - x1 + y2 - y1) // 2)
        cv2.circle(heatmap, (cx, cy), radius, 1.0, -1)
    
    # Blur for smooth heatmap
    heatmap = cv2.GaussianBlur(heatmap, (99, 99), 0)
    
    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Overlay on original image
    overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    
    return overlay


def count_people(image_bytes):
    """
    Detect and count people in an image, with density estimation.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        dict with keys:
            - total_people: count of detected people
            - density: crowd density level string
            - detections: list of person detections
            - annotated_image: base64 encoded annotated image
            - heatmap_image: base64 encoded density heatmap
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Could not decode image"}
    
    model = get_model()
    
    # Auto-detect device
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results = model(img, verbose=False, device=device)[0]
    
    annotated = img.copy()
    detections = []
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id == PERSON_CLASS_ID:
            confidence = float(box.conf[0])
            if confidence < 0.35:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            detections.append({
                'confidence': round(confidence, 3),
                'bbox': [x1, y1, x2, y2]
            })
            
            # Draw detection
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 255), 2)
            label = f"Person {confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), (0, 220, 255), -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Crowd density
    image_area = img.shape[0] * img.shape[1]
    density = estimate_density(len(detections), image_area)
    
    # Draw counter badge
    count_text = f"People: {len(detections)} | Density: {density}"
    cv2.rectangle(annotated, (5, 5), (len(count_text) * 14 + 10, 40), (0, 0, 0), -1)
    cv2.putText(annotated, count_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    
    # Generate heatmap
    heatmap = generate_heatmap(img, detections)
    
    # Encode images
    _, buf1 = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    annotated_b64 = base64.b64encode(buf1).decode('utf-8')
    
    _, buf2 = cv2.imencode('.jpg', heatmap, [cv2.IMWRITE_JPEG_QUALITY, 90])
    heatmap_b64 = base64.b64encode(buf2).decode('utf-8')
    
    return {
        'total_people': len(detections),
        'density': density,
        'detections': detections,
        'annotated_image': annotated_b64,
        'heatmap_image': heatmap_b64
    }
