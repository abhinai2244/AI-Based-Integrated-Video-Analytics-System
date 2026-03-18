"""
Helmet Detection Module
Uses dual YOLOv8 models to detect persons, motorcycles, and helmet compliance.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import base64
import os
import torch

# Models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'helmet')
PRIMARY_MODEL_PATH = os.path.join(MODELS_DIR, 'yolov8n.pt')
HELMET_MODEL_PATH = os.path.join(MODELS_DIR, 'helmet_model.pt')

# Cache for models
_primary_model = None
_helmet_model = None

def get_models():
    global _primary_model, _helmet_model
    if _primary_model is None:
        _primary_model = YOLO(PRIMARY_MODEL_PATH)
    if _helmet_model is None:
        _helmet_model = YOLO(HELMET_MODEL_PATH)
    return _primary_model, _helmet_model

def detect_helmets(image_bytes=None, img=None):
    """
    Detect persons, motorcycles, and helmet status.
    """
    if img is None:
        if image_bytes is None:
            return {"error": "No image data provided"}
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Could not decode image"}

    primary_model, helmet_model = get_models()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Run Primary inference
    primary_results = primary_model(img, verbose=False, device=device)[0]
    # Run Helmet inference
    helmet_results = helmet_model(img, verbose=False, device=device)[0]

    annotated = img.copy()
    detections = []
    
    # Helmet status counts
    status_counts = {"With helmet": 0, "Without helmet": 0}
    
    # Track relevant primary objects
    primary_objects = {
        'person': 0,
        'motorcycle': 0,
        'bicycle': 0
    }

    # Annotate Primary Results (Person, Motorcycle, Bicycle)
    for box in primary_results.boxes:
        cls_id = int(box.cls[0])
        label = primary_model.names[cls_id]
        if label in primary_objects:
            primary_objects[label] += 1
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Draw primary box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            detections.append({
                'class': label,
                'confidence': round(conf, 2),
                'bbox': [x1, y1, x2, y2]
            })

    # Annotate Helmet Results
    for box in helmet_results.boxes:
        cls_id = int(box.cls[0])
        label = helmet_model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Normalize labels (handle potential space issues)
        clean_label = label.strip()
        if "With" in clean_label and "Without" not in clean_label:
            status_counts["With helmet"] += 1
            color = (0, 255, 0) # Green
        else:
            status_counts["Without helmet"] += 1
            color = (0, 0, 255) # Red
            
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{clean_label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        detections.append({
            'class': clean_label,
            'confidence': round(conf, 2),
            'bbox': [x1, y1, x2, y2],
            'is_violation': "Without" in clean_label
        })

    # Summary Overlay
    overlay = annotated.copy()
    cv2.rectangle(overlay, (10, 10), (250, 150), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
    
    cv2.putText(annotated, "HELMET ANALYSIS", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated, f"Riders: {primary_objects['person']}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(annotated, f"With Helmet: {status_counts['With helmet']}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(annotated, f"Violations: {status_counts['Without helmet']}", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    _, buffer = cv2.imencode('.jpg', annotated)
    annotated_b64 = base64.b64encode(buffer).decode('utf-8')

    return {
        'total_riders': primary_objects['person'],
        'violations': status_counts['Without helmet'],
        'detections': detections,
        'annotated_image': annotated_b64,
        'status': 'WARNING' if status_counts['Without helmet'] > 0 else 'CLEAR'
    }
