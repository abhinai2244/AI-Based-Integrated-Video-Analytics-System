"""
Automatic Number Plate Recognition (ANPR) Module
Uses fine-tuned YOLOv8 (best.pt) for plate detection + EasyOCR for license plate reading.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import base64
import re
import json
import os
from security_utils import send_alert_email

# Models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, 'best.pt')

# ── Global model handles ──
_yolo_model = None
_ocr_reader = None  # EasyOCR is now the primary OCR

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(YOLO_MODEL_PATH)
    return _yolo_model

def get_ocr_reader():
    """Primary OCR: EasyOCR reader."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        import torch
        use_gpu = torch.cuda.is_available()
        _ocr_reader = easyocr.Reader(['en'], gpu=use_gpu, model_storage_directory=MODELS_DIR)
    return _ocr_reader

def read_plate(crop_bgr):
    """
    Read license plate using EasyOCR.
    """
    return easyocr_read_plate(crop_bgr)


def easyocr_read_plate(crop_bgr):
    """Fallback: use EasyOCR to read license plate from a crop."""
    reader = get_ocr_reader()
    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    
    # Preprocess
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filter to remove noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 9, 15, 15)
    
    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    
    # Upscale small crops
    h, w = enhanced.shape[:2]
    if h < 80:
        scale = 80 / max(1, h)
        enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Use beamsearch for better character prediction
    ocr_results = reader.readtext(enhanced, allowlist=allowlist, decoder='beamsearch')
    
    # Combine all detected text, filtering out noise
    valid_texts = []
    for (bbox_pts, text, conf) in ocr_results:
        clean = re.sub(r'[^A-Z0-9]', '', text).upper()
        # Remove typical Indian plate noise ("IND")
        clean = clean.replace('IND', '')
        
        if len(clean) >= 2 and conf > 0.3:
            valid_texts.append((clean, conf))
    
    if not valid_texts:
        return None, 0.0
    
    # Try combined text first
    combined = "".join([t[0] for t in valid_texts])
    avg_conf = sum([t[1] for t in valid_texts]) / len(valid_texts)
    
    if is_valid_plate_text(combined):
        print(f"DEBUG: OCR Decoded Plate -> {combined} (Conf: {avg_conf:.2f})")
        return combined, avg_conf
    
    # Try individual longest best piece
    longest = max(valid_texts, key=lambda x: len(x[0]))
    if is_valid_plate_text(longest[0]):
        print(f"DEBUG: OCR Decoded Plate (Longest part) -> {longest[0]} (Conf: {longest[1]:.2f})")
        return longest[0], longest[1]
    
    return None, 0.0


def is_indian_plate(text):
    """Validate Indian license plate format."""
    clean = re.sub(r'\s+', '', text).upper()
    if len(clean) >= 2:
        state_part = clean[:2].replace('0', 'O').replace('1', 'I')
        clean = state_part + clean[2:]
    indian_regex = r'^[A-Z]{2}[0-9OIZS]{1,2}[A-Z0-9]{0,3}[0-9OIZS]{3,4}$'
    return re.match(indian_regex, clean) is not None


def is_valid_plate_text(text):
    """Check if text looks like a real plate number."""
    if len(text) < 5 or len(text) > 15:
        return False
    has_letters = any(c.isalpha() for c in text)
    has_digits = any(c.isdigit() for c in text)
    return has_letters and has_digits


def get_blacklist():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config_data.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
                return set(data.get("blacklist_plates", []))
        except Exception:
            return set()
    return set()


def detect_plates(image_bytes=None, img=None, high_throughput=False):
    """
    Detect license plates using:
    1. YOLO vehicle detection → multi-region crop
    2. Awiros ANPR-OCR for text reading (98.42% accuracy on Indian plates)
    3. EasyOCR fallback if Awiros not available (Skipped in High Throughput mode)
    4. Contour-based detection as last resort
    """
    if img is None:
        if image_bytes is None:
            return {"error": "No image data provided"}
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Could not decode image"}
    
    # Use a faster inference size if high_throughput
    inf_size = 480 if high_throughput else 640

    annotated = img.copy()
    plates = []
    seen_texts = set()
    blacklist = get_blacklist()
    
    def add_plate(text, conf, x1, y1, x2, y2):
        """Helper to add validated plate."""
        if text in seen_texts:
            return
        if not is_valid_plate_text(text):
            return
        seen_texts.add(text)
        is_ind = is_indian_plate(text)
        is_blk = text in blacklist
        plates.append({
            'text': text, 'confidence': round(float(conf), 3),
            'is_indian_standard': is_ind, 'is_blacklisted': is_blk,
            'bbox': [x1, y1, x2, y2]
        })
        color = (0, 0, 255) if is_blk else ((0, 255, 0) if is_ind else (0, 165, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        label = f"{'ALERT ' if is_blk else ''}{'IND: ' if is_ind else ''}{text}"
        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Trigger email alert for blacklisted plates
        if is_blk:
            send_alert_email(
                subject=f"PLATE ALERT: Blacklisted Plate {text} Detected",
                message=f"Crucial Alert: Number plate '{text}' on the blacklist has been identified.",
                alert_type="plate"
            )

    def read_plate_from_crop(crop_bgr):
        """Read plate text using EasyOCR."""
        return easyocr_read_plate(crop_bgr)
    
    # ── Strategy 1: YOLO License Plate Detection → OCR ──
    try:
        model = get_yolo_model()
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        results = model(img, verbose=False, device=device)[0]
        
        for box in results.boxes:
            if float(box.conf[0]) > 0.3:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Add a small padding for better OCR results
                pad = 5
                crop = img[max(0, y1-pad):min(img.shape[0], y2+pad), 
                           max(0, x1-pad):min(img.shape[1], x2+pad)]
                
                if crop.size == 0: continue
                
                text, conf = read_plate_from_crop(crop)
                if text and is_valid_plate_text(text):
                    add_plate(text, conf, x1, y1, x2, y2)
                        
    except Exception as e:
        print(f"YOLO ANPR detection error: {e}")
    
    # ── Strategy 2: Contour-based plate region detection (Skipped in High Throughput) ──
    if not plates and not high_throughput:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200)
        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
        
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if 1.5 <= aspect_ratio <= 6.0 and w > 50 and h > 12:
                    pad = 8
                    roi = img[max(0, y-pad):min(img.shape[0], y+h+pad),
                              max(0, x-pad):min(img.shape[1], x+w+pad)]
                    if roi.size == 0:
                        continue
                    text, conf = read_plate_from_crop(roi)
                    if text and is_valid_plate_text(text):
                        add_plate(text, conf, x, y, x+w, y+h)
    
    # ── Strategy 3: Full image OCR (Skipped in High Throughput) ──
    if not plates and not high_throughput:
        try:
            reader = get_ocr_reader()
            allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
            ocr_all = reader.readtext(img, allowlist=allowlist)
            for (bbox_pts, text, conf) in ocr_all:
                clean = re.sub(r'[^A-Z0-9]', '', text).upper()
                if len(clean) >= 6 and conf > 0.35 and is_valid_plate_text(clean):
                    if is_indian_plate(clean) or conf > 0.4:
                        pts = np.array(bbox_pts, dtype=np.int32)
                        x_min, y_min = pts.min(axis=0)
                        x_max, y_max = pts.max(axis=0)
                        add_plate(clean, conf, int(x_min), int(y_min), int(x_max), int(y_max))
        except Exception:
            pass

    _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {
        'plates': plates,
        'annotated_image': base64.b64encode(buffer).decode('utf-8'),
        'total_plates': len(plates)
    }
