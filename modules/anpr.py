"""
Automatic Number Plate Recognition (ANPR) Module
Optimized for Indian High-Security Registration Plates (HSRP).
"""

import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import base64
import re

# Load models globally
_yolo_model = None
_ocr_reader = None

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO('yolov8l.pt')  # Upgraded to Large model
    return _yolo_model

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        # Enable GPU for EasyOCR if available
        import torch
        use_gpu = torch.cuda.is_available()
        _ocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
    return _ocr_reader

def is_indian_plate(text):
    """
    Validate if the text follows Indian license plate standards.
    Regex: [State Code][District Code][Optional Alpha][Unique Number]
    Example: MH12AB1234, DL3C1234, KA011234
    """
    clean = re.sub(r'\s+', '', text).upper()
    # AA NN (A-ZZZ) NNNN
    indian_regex = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$'
    return re.match(indian_regex, clean) is not None

def find_plate_contours(img):
    """Contour detection for potential plate regions."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    
    plate_regions = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 1.5 <= aspect_ratio <= 6.0 and w > 50 and h > 12:
                plate_regions.append((x, y, w, h))
    return plate_regions

def detect_plates(image_bytes):
    """Detect plates with Indian optimization."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return {"error": "Could not decode image"}
    
    annotated = img.copy()
    reader = get_ocr_reader()
    plates = []
    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    
    # 1. Shape Detection Strategy
    regions = find_plate_contours(img)
    
    for (x, y, w, h) in regions:
        pad = 8
        roi = img[max(0, y-pad):min(img.shape[0], y+h+pad), 
                  max(0, x-pad):min(img.shape[1], x+w+pad)]
        if roi.size == 0: continue
        
        # Pre-process ROI
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        ocr_results = reader.readtext(gray, allowlist=allowlist)
        combined_text = "".join([re.sub(r'[^A-Z0-9]', '', r[1]).upper() for r in ocr_results])
        max_conf = max([r[2] for r in ocr_results]) if ocr_results else 0
        
        if combined_text:
            is_ind = is_indian_plate(combined_text)
            if is_ind or (len(combined_text) >= 7 and max_conf > 0.4):
                plates.append({
                    'text': combined_text,
                    'confidence': round(float(max_conf), 3),
                    'is_indian_standard': is_ind,
                    'bbox': [x, y, x + w, y + h]
                })
                color = (0, 255, 0) if is_ind else (0, 165, 255)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
                label = f"{'IND: ' if is_ind else ''}{combined_text}"
                cv2.putText(annotated, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 2. Fallback: Full Scrape for Indian patterns only
    if not plates:
        ocr_all = reader.readtext(img, allowlist=allowlist)
        for (bbox, text, conf) in ocr_all:
            clean = re.sub(r'[^A-Z0-9]', '', text).upper()
            if is_indian_plate(clean) and conf > 0.3:
                pts = np.array(bbox, dtype=np.int32)
                x_min, y_min = pts.min(axis=0)
                x_max, y_max = pts.max(axis=0)
                plates.append({
                    'text': clean, 'confidence': round(float(conf), 3),
                    'is_indian_standard': True, 'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)]
                })
                cv2.rectangle(annotated, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                cv2.putText(annotated, f"IND: {clean}", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {
        'plates': plates,
        'annotated_image': base64.b64encode(buffer).decode('utf-8'),
        'total_plates': len(plates)
    }
