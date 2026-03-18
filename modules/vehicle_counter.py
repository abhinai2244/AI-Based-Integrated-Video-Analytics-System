"""
Vehicle Counting & Classification Module
Uses YOLOv8 Nano for real-time vehicle detection and classification.
Supports: car, motorcycle, bus, truck
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from PIL import Image
import io
import base64
import os

# Models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, 'yolov8l.pt')

# Speed tracker state
_tracker_history = {}  # {id: {'pt': (x,y), 'time': t}}
# Assuming 30 FPS camera, ~12 pixels = 1 meter context
PIXELS_PER_METER = 12.0

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
        _model = YOLO(YOLO_MODEL_PATH)
    return _model


def detect_vehicles(image_bytes=None, img=None, high_throughput=False):
    """
    Detect and classify vehicles in an image.
    
    Args:
        image_bytes: Raw image bytes (optional if img is provided)
        img: Direct numpy image (optional if image_bytes is provided)
        high_throughput: If True, optimize for speed over accuracy
        
    Returns:
        dict with keys:
            - counts: dict of vehicle type -> count
            - detections: list of detection dicts (class, confidence, bbox)
            - annotated_image: base64 encoded annotated image
            - total: total vehicle count
    """
    # Decode image if not provided
    if img is None:
        if image_bytes is None:
            return {"error": "No image data provided"}
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Could not decode image"}
    
    model = get_model()
    
    # Auto-detect device (GPU if available)
    # Auto-detect device (GPU if available)
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Run standard detection (no track)
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
    
    current_time = time.time()
    
    # Collect all centroids for this frame
    current_centroids = []
    current_boxes = []
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in VEHICLE_CLASSES:
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            current_centroids.append((cx, cy))
            current_boxes.append({
                'cls': VEHICLE_CLASSES[cls_id],
                'conf': confidence,
                'bbox': [x1, y1, x2, y2],
                'cx': cx, 'cy': cy
            })
            
            counts[VEHICLE_CLASSES[cls_id]] += 1
            
    # Simple Centroid Tracking
    from scipy.spatial import distance
    
    assigned_this_frame = {}
    
    if len(current_centroids) > 0:
        if len(_tracker_history) == 0:
            # First frame or all tracks lost: assign new IDs
            for i, box_data in enumerate(current_boxes):
                new_id = int(current_time * 1000) + i  # Generate unique ID
                assigned_this_frame[new_id] = box_data
                _tracker_history[new_id] = {'pt': (box_data['cx'], box_data['cy']), 'time': current_time, 'speed': 0}
        else:
            # Match existing tracks
            object_ids = list(_tracker_history.keys())
            object_pts = [v['pt'] for v in _tracker_history.values()]
            
            # Distance matrix between old points and new points
            D = distance.cdist(np.array(object_pts), np.array(current_centroids))
            
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                # Check if distance is reasonably small (e.g. max 150 pixels movement between frames)
                if D[row, col] > 150:
                    continue
                    
                obj_id = object_ids[row]
                box_data = current_boxes[col]
                
                # Calculate speed
                prev_pt = _tracker_history[obj_id]['pt']
                prev_t = _tracker_history[obj_id]['time']
                dt = current_time - prev_t
                
                speed_kmh = _tracker_history[obj_id].get('speed', 0)
                if dt > 0 and dt < 2.0: # Only update speed if dt is reasonable
                    dist_px = np.sqrt((box_data['cx'] - prev_pt[0])**2 + (box_data['cy'] - prev_pt[1])**2)
                    dist_m = dist_px / PIXELS_PER_METER
                    speed_mps = dist_m / dt
                    # Smooth speed calculation
                    new_speed = speed_mps * 3.6
                    if speed_kmh == 0:
                        speed_kmh = new_speed
                    else:
                        speed_kmh = speed_kmh * 0.5 + new_speed * 0.5
                
                _tracker_history[obj_id] = {'pt': (box_data['cx'], box_data['cy']), 'time': current_time, 'speed': speed_kmh}
                assigned_this_frame[obj_id] = box_data
                
                used_rows.add(row)
                used_cols.add(col)
                
            # Register new objects
            unused_cols = set(range(len(current_centroids))) - used_cols
            for col in unused_cols:
                box_data = current_boxes[col]
                new_id = int(current_time * 1000) + col
                assigned_this_frame[new_id] = box_data
                _tracker_history[new_id] = {'pt': (box_data['cx'], box_data['cy']), 'time': current_time, 'speed': 0}
                
    # Draw and format results
    for obj_id, data in assigned_this_frame.items():
        vehicle_type = data['cls']
        confidence = data['conf']
        x1, y1, x2, y2 = data['bbox']
        speed_kmh = _tracker_history[obj_id].get('speed', 0)
        
        detections.append({
            'id': int(obj_id),
            'class': vehicle_type,
            'confidence': round(float(confidence), 3),
            'speed': round(float(speed_kmh), 1) if speed_kmh > 0 else None,
            'bbox': [int(x) for x in data['bbox']]
        })

        
        # Draw bounding box
        color = colors.get(vehicle_type, (0, 255, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{vehicle_type} {confidence:.2f}"
        if speed_kmh > 0:
            label += f" | {speed_kmh:.1f} km/h"
            
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # ── Summary HUD Overlay ───────────────────────────────────────────────
    overlay = annotated.copy()
    hud_h, hud_w = 40 + (len(counts) * 28), 240
    cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (30, 30, 30), -1)
    cv2.rectangle(annotated, (10, 10), (10 + hud_w, 10 + hud_h), (0, 255, 204), 2) # Border
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

    cv2.putText(annotated, "VEHICLE SUMMARY", (25, 38), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 204), 2)
    
    y_off = 72
    for vt, count in counts.items():
        v_color = colors.get(vt, (255, 255, 255))
        cv2.putText(annotated, f"{vt.upper()}:", (25, y_off), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(annotated, str(count), (180, y_off), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, v_color, 2)
        y_off += 28
    # ──────────────────────────────────────────────────────────────────────
    
    # Cleanup old tracker histories (memory management)
    expired = [k for k, v in list(_tracker_history.items()) if current_time - v['time'] > 2.0]
    for k in expired:
        del _tracker_history[k]
    
    # Encode annotated image to base64
    _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    annotated_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        'counts': counts,
        'detections': detections,
        'annotated_image': annotated_b64,
        'total': sum(counts.values())
    }
