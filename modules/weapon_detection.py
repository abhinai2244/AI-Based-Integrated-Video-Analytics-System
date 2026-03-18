"""
Weapon Detection Module v2
==========================
Uses YOLOv8 for real-time weapon detection (pistols, rifles, knives, etc.).
- Lower confidence threshold for better recall
- Accepts all classes from custom model
- Fallback: run with imsz=1280 for small object detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
import base64
import os
import time
from security_utils import send_alert_email

# Models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
YOLO_WEAPON_MODEL = os.path.join(MODELS_DIR, 'weapon_v2.pt')

_model = None

# Confidence threshold — lower = more detections (better recall, slight more false +)
CONF_THRESHOLD = 0.30   # was 0.5 — too high for custom model, missed weapons

# Draw colours per class (BGR)
CLASS_COLORS = {
    0: (0, 60, 255),    # guns   → vivid red-orange
    1: (0, 140, 255),   # knife  → orange
}
DEFAULT_COLOR = (0, 0, 220)


def get_model():
    global _model
    if _model is None:
        if os.path.exists(YOLO_WEAPON_MODEL):
            _model = YOLO(YOLO_WEAPON_MODEL)
            print(f"[Weapon] Loaded custom model: {YOLO_WEAPON_MODEL}")
        else:
            # Fallback to standard YOLOv8s — won't detect custom weapons well
            _model = YOLO('yolov8s.pt')
            print("[Weapon] WARNING: weapon_v2.pt not found, using yolov8s.pt fallback")
    return _model


def detect_weapons(image_bytes=None, img=None, high_throughput=False):
    """
    Detect weapons in an image.
    Args:
        image_bytes: raw JPEG bytes
        img: numpy array (BGR)
        high_throughput: if True, use smaller imgsz for speed
    """
    if img is None:
        if image_bytes is None:
            return {"error": "No image data provided"}
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Could not decode image"}

    model = get_model()

    # Use high resolution for better small-object detection
    imgsz = 640 if high_throughput else 1280

    # Run detection
    results = model(img, verbose=False, conf=0.20, imgsz=imgsz)[0] # Lowered from 0.30

    annotated = img.copy()
    detections = []

    names = results.names  # class id → name from the model

    for box in results.boxes:
        conf = float(box.conf[0])
        # Model-level filtering is usually enough, but we manually check again
        if conf < 0.20:
            continue

        cls_id = int(box.cls[0])
        raw_name = names.get(cls_id, f'weapon_{cls_id}').lower()
        
        # Professional Mapping for labels
        mapping = {
            'guns': 'FIREARM',
            'gun': 'FIREARM',
            'rifle': 'RIFLE',
            'pistol': 'PISTOL',
            'knife': 'BLADE',
            'blade': 'BLADE'
        }
        cls_name = mapping.get(raw_name, raw_name.upper())

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Add more padding to the detection box for better view of the full weapon
        pad_x = 20
        pad_y = 15
        x1, y1 = max(0, x1-pad_x), max(0, y1-pad_y)
        x2, y2 = min(img.shape[1], x2+pad_x), min(img.shape[0], y2+pad_y)

        detections.append({
            'class': cls_name,
            'confidence': round(conf, 3),
            'bbox': [x1, y1, x2, y2]
        })

        # Drawing
        color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
        thickness = 3
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Label background
        label = f"ALERT: {cls_name.upper()} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        
        print(f"[Weapon] Found: {cls_name} with {conf:.2f} confidence at [{x1},{y1},{x2},{y2}]")

    # ── HUD banner ───────────────────────────────────────────────────────────
    h, w = img.shape[:2]
    if detections:
        weapon_names = ", ".join(set(d['class'].upper() for d in detections))
        banner = f"WEAPON DETECTED: {weapon_names} ({len(detections)})"
        cv2.rectangle(annotated, (0, 0), (w, 38), (0, 0, 180), -1)
        cv2.putText(annotated, banner, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(annotated, (0, 0), (w, 38), (0, 120, 0), -1)
        cv2.putText(annotated, "Status: Clear — No Weapons Detected", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 2, cv2.LINE_AA)

    # Encode
    _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    annotated_b64 = base64.b64encode(buffer).decode('utf-8')

    # Send alert email
    if detections:
        weapons_list = ", ".join(d['class'].upper() for d in detections)
        send_alert_email(
            subject=f"WEAPON ALERT: {weapons_list} Detected",
            message=f"Critical Alert: {len(detections)} weapon(s) detected: {weapons_list}. Immediate action required.",
            alert_type="weapon"
        )

    return {
        'total_weapons': len(detections),
        'detections': detections,
        'annotated_image': annotated_b64,
        'status': 'alert' if detections else 'clear'
    }
