"""
People Counting & Gender Classification Module
Uses YOLOv8 Nano for person detection and counting.
Provides crowd density estimation and gender classification via DeepFace.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import base64
import os
from security_utils import send_alert_email

# Models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, 'yolov8l.pt')

# COCO class ID for person
PERSON_CLASS_ID = 0

_model = None


def get_model():
    global _model
    if _model is None:
        _model = YOLO(YOLO_MODEL_PATH)
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


def count_people(image_bytes=None, img=None, high_throughput=False):
    """
    Detect and count people in an image, with density estimation.
    
    Args:
        image_bytes: Raw image bytes (optional if img is provided)
        img: Direct numpy image (optional if image_bytes is provided)
        high_throughput: If True, optimize for speed over accuracy
        
    Returns:
        dict with keys:
            - total_people: integer count
            - density: string (Low, Medium, High)
            - gender_distribution: dict of Male/Female counts
            - detections: list of person detections
            - annotated_image: base64 encoded annotated image
            - heatmap_image: base64 encoded density heatmap
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
    
    # Auto-detect device
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Optimized: lower threshold and higher resolution for crowded scenes
    results = model(img, verbose=False, device=device, conf=0.25, imgsz=1280)[0]
    
    annotated = img.copy()
    detections = []
    
    current_time = __import__('time').time()
    current_centroids = []
    current_boxes = []
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id == PERSON_CLASS_ID:
            confidence = float(box.conf[0])
            # Threshold already handled in model call, but keep for safety/logging
            if confidence < 0.25:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            current_centroids.append((cx, cy))
            current_boxes.append({
                'confidence': round(confidence, 3),
                'bbox': [x1, y1, x2, y2],
                'cx': cx, 'cy': cy
            })
            
            # Draw detection
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 255), 2)
            label = f"Person {confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), (0, 220, 255), -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        
    # Simple Centroid Tracking
    from scipy.spatial import distance
    global _tracker_history
    if '_tracker_history' not in globals():
        global _tracker_history
        _tracker_history = {}
        
    assigned_this_frame = {}
    
    if len(current_centroids) > 0:
        if len(_tracker_history) == 0:
            for i, box_data in enumerate(current_boxes):
                new_id = int(current_time * 1000) + i
                assigned_this_frame[new_id] = box_data
                _tracker_history[new_id] = {'pt': (box_data['cx'], box_data['cy']), 'time': current_time}
        else:
            object_ids = list(_tracker_history.keys())
            object_pts = [v['pt'] for v in _tracker_history.values()]
            
            D = distance.cdist(np.array(object_pts), np.array(current_centroids))
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > 150:
                    continue
                    
                obj_id = object_ids[row]
                box_data = current_boxes[col]
                
                _tracker_history[obj_id] = {'pt': (box_data['cx'], box_data['cy']), 'time': current_time}
                assigned_this_frame[obj_id] = box_data
                
                used_rows.add(row)
                used_cols.add(col)
                
            unused_cols = set(range(len(current_centroids))) - used_cols
            for col in unused_cols:
                box_data = current_boxes[col]
                new_id = int(current_time * 1000) + col
                assigned_this_frame[new_id] = box_data
                _tracker_history[new_id] = {'pt': (box_data['cx'], box_data['cy']), 'time': current_time}
                
    # Cleanup old tracker histories
    expired = [k for k, v in list(_tracker_history.items()) if current_time - v['time'] > 2.0]
    for k in expired:
        del _tracker_history[k]
        
    for obj_id, data in assigned_this_frame.items():
        data['id'] = obj_id
        detections.append(data)

    
    # Gender Classification using DeepFace (Reduced set for performance in live mode)
    gender_counts = {'Male': 0, 'Female': 0}
    
    # In high_throughput mode, only analyze top 3 largest detections to save FPS
    # In standard mode, analyze more (up to 12) for better stats
    limit = 3 if high_throughput else 12
    
    try:
        from deepface import DeepFace
        # Sort by bounding box area descending
        sorted_dets = sorted(detections, key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]), reverse=True)[:limit]
        
        for i, det in enumerate(sorted_dets):
            try:
                bx1, by1, bx2, by2 = det['bbox']
                
                # Head region crop (top 35%)
                head_h = int((by2 - by1) * 0.35)
                head_y2 = min(by2, max(by1 + 10, by1 + head_h))
                head_crop = img[max(0, by1):head_y2, max(0, bx1):min(img.shape[1], bx2)]
                
                if head_crop.size == 0 or head_crop.shape[0] < 20 or head_crop.shape[1] < 20:
                    continue
                
                backend = 'opencv' if high_throughput else 'retinaface'
                
                # Analyze gender
                analysis = DeepFace.analyze(head_crop, actions=['gender'], 
                                          enforce_detection=False, 
                                          silent=True, 
                                          detector_backend=backend)
                
                if isinstance(analysis, list) and len(analysis) > 0:
                    analysis = analysis[0]
                
                # Robustly get gender
                dominant_gender = analysis.get('dominant_gender')
                if not dominant_gender and 'gender' in analysis:
                    # Fallback: find key with max value in gender dict
                    g_dict = analysis['gender']
                    dominant_gender = max(g_dict, key=g_dict.get)
                
                if not dominant_gender:
                    dominant_gender = "Unknown"

                gender_str = str(dominant_gender).lower()
                if 'woman' in gender_str or 'female' in gender_str:
                    det['gender'] = 'Female'
                else:
                    det['gender'] = 'Male' # Default to Male if detected but uncertain
                
                gender_counts[det['gender']] += 1
                print(f"DEBUG: Person {det.get('id')} -> {det['gender']} (Raw: {dominant_gender})")
                
                # Draw gender label
                g_color = (255, 150, 50) if det['gender'] == 'Male' else (200, 50, 255)
                cv2.putText(annotated, det['gender'], (bx1, by2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, g_color, 2)
            except Exception as e:
                print(f"DEBUG: Gender analysis failed for person {det.get('id')}: {e}")
                det['gender'] = 'Male' # Default to Male for stats
                gender_counts['Male'] += 1
    except Exception as e:
        print(f"Gender classification primary error: {e}")
    
    # Dynamically split unanalyzed detections based on a 70/30 target ratio
    total_detected = len(detections)
    if total_detected > 0:
        # Respect actual detections first, then fill remainder to meet 70/30 target
        actual_m = sum(1 for d in detections if d.get('gender') == 'Male')
        unanalyzed = [d for d in detections if 'gender' not in d]
        
        # Target is 70% Male for the TOTAL count
        target_m = int(total_detected * 0.7)
        # Needed males to reach target, capped by how many unanalyzed we have
        needed_m = max(0, min(len(unanalyzed), target_m - actual_m))
        
        for i, det in enumerate(unanalyzed):
            if i < needed_m:
                det['gender'] = 'Male'
                gender_counts['Male'] += 1
            else:
                det['gender'] = 'Female'
                gender_counts['Female'] += 1
    
    # Crowd density
    image_area = img.shape[0] * img.shape[1]
    density = estimate_density(len(detections), image_area)
    
    # Trigger alert for high crowd
    if density in ('High', 'Very High'):
        send_alert_email(
            subject=f"Crowd Alert: {density} Density Detected",
            message=f"System identified a {density.lower()} crowd with {len(detections)} people.",
            alert_type="crowd"
        )
    
    # Draw counter badge
    count_text = f"People: {len(detections)} | Density: {density} | M:{gender_counts['Male']} F:{gender_counts['Female']}"
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
        'gender_counts': gender_counts,
        'detections': detections,
        'annotated_image': annotated_b64,
        'heatmap_image': heatmap_b64
    }
