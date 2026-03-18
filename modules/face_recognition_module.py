"""
Face Recognition Module - Robust Version 3.0
===========================================
A clean, rebuilt implementation from scratch focused on reliability and accuracy.
Features: Multi-image averaging, robust detection backends, temporal confirmation, and evidence capture.
"""

import cv2
import numpy as np
from deepface import DeepFace
import base64
import os
from security_utils import send_alert_email
import time
import pickle
import traceback
from datetime import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# ── CONFIGURATION ──────────────────────────────────────────────
# Configurable via environment variables or defaults
COSINE_THRESHOLD = float(os.environ.get('FACE_COSINE_THRESHOLD', '0.40'))
MIN_FACE_SIZE = int(os.environ.get('FACE_MIN_SIZE', '60'))
BLUR_THRESHOLD = float(os.environ.get('FACE_BLUR_THRESHOLD', '80.0'))
TEMPORAL_CONFIRM_FRAMES = int(os.environ.get('FACE_TEMPORAL_FRAMES', '2'))
MODEL_NAME = os.environ.get('FACE_MODEL_NAME', 'Facenet512')
DETECTOR_BACKEND = os.environ.get('FACE_DETECTOR', 'yolov8') # Use YOLOv8 for detection
FAST_DETECTOR = 'opencv' 

# Models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
YOLO_FACE_MODEL = os.path.join(MODELS_DIR, 'yolov8l.pt') # Using yolov8s as requested

# ── PATHS ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
BLACKLIST_DIR = os.path.join(BASE_DIR, 'face_blacklist_db')
WHITELIST_DIR = os.path.join(BASE_DIR, 'whitelist_db')
ALERTS_DIR = os.path.join(BASE_DIR, 'alerts')
ALERTS_LOG = os.path.join(BASE_DIR, 'alerts_log.txt')
EMBEDDINGS_CACHE = os.path.join(BASE_DIR, 'face_embeddings_cache.pkl')

os.makedirs(ALERTS_DIR, exist_ok=True)
os.makedirs(BLACKLIST_DIR, exist_ok=True)
os.makedirs(WHITELIST_DIR, exist_ok=True)

# ── STATE ──────────────────────────────────────────────────────
_precomputed_blacklist = {}   # {name: normalized_avg_embedding}
_precomputed_whitelist = {}
_db_loaded = False
_tracker = None
_track_history = {}           # {track_id: {"name": str, "count": int, "alert_sent": bool}}
_track_embeddings = {}        # {track_id: embedding}
_yolo_model = None

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        # Load yolov8s.pt - it will auto-download to the project folder if missing
        _yolo_model = YOLO(YOLO_FACE_MODEL)
    return _yolo_model

# ═══════════════════════════════════════════════════════════════
# 1. DATABASE & EMBEDDING CORE
# ═══════════════════════════════════════════════════════════════

def _generate_embedding(img):
    """Generate a FaceNet embedding for a face image."""
    try:
        results = DeepFace.represent(
            img, model_name=MODEL_NAME, 
            enforce_detection=False, 
            detector_backend='skip' # Already localized
        )
        if results:
            emb = np.array(results[0]['embedding'])
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 1e-6 else emb
    except Exception:
        pass
    return None

def initialize_databases(force_refresh=False):
    """Load and process images from blacklist and whitelist folders."""
    global _precomputed_blacklist, _precomputed_whitelist, _db_loaded
    
    if _db_loaded and not force_refresh:
        return

    # Automatic Cache Refresh: Check if directory content has changed
    def get_dir_hash(path):
        if not os.path.exists(path): return ""
        files = []
        for root, dirs, filenames in os.walk(path):
            for f in filenames:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    files.append(f + str(os.path.getmtime(os.path.join(root, f))))
        return str(hash(tuple(sorted(files))))

    current_hash = get_dir_hash(BLACKLIST_DIR) + get_dir_hash(WHITELIST_DIR)
    
    # Try loading cache
    if not force_refresh and os.path.exists(EMBEDDINGS_CACHE):
        try:
            with open(EMBEDDINGS_CACHE, 'rb') as f:
                data = pickle.load(f)
                if data.get('hash') == current_hash:
                    _precomputed_blacklist = data.get('blacklist', {})
                    _precomputed_whitelist = data.get('whitelist', {})
                    _db_loaded = True
                    print(f"[FaceDB] Loaded from cache: {len(_precomputed_blacklist)} blacklist, {len(_precomputed_whitelist)} whitelist")
                    return
                else:
                    print("[FaceDB] Database changed, regenerating cache...")
        except:
             pass

    print("[FaceDB] Generating fresh embeddings...")
    
    # Pre-load YOLO for detect_backend='yolov8' if DeepFace supports it
    # But for initialization from files, we'll use a standard backend or yolov8 if stable
    
    def process_dir(db_path):
        db = {}
        if not os.path.exists(db_path): return db
        
        # Check if directory structure has changed since last cache
        dir_content_hash = str(sorted(os.listdir(db_path)))
        
        for entry in os.listdir(db_path):
            path = os.path.join(db_path, entry)
            person_name = entry
            embs = []
            
            # Handle Subdirectory (Structure A: Person/img.jpg)
            if os.path.isdir(path):
                for f in os.listdir(path):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(path, f)
                        emb = _generate_embedding_from_file(img_path)
                        if emb is not None: embs.append(emb)
            # Handle Single File (Structure B: Person.jpg)
            elif entry.lower().endswith(('.jpg', '.jpeg', '.png')):
                person_name = os.path.splitext(entry)[0]
                emb = _generate_embedding_from_file(path)
                if emb is not None: embs.append(emb)
            
            if embs:
                avg_emb = np.mean(embs, axis=0)
                norm = np.linalg.norm(avg_emb)
                db[person_name] = avg_emb / norm if norm > 1e-6 else avg_emb
                print(f"[FaceDB] Processed {person_name}: {len(embs)} images")
        return db

    _precomputed_blacklist = process_dir(BLACKLIST_DIR)
    _precomputed_whitelist = process_dir(WHITELIST_DIR)

    # Save cache
    try:
        with open(EMBEDDINGS_CACHE, 'wb') as f:
            pickle.dump({
                'blacklist': _precomputed_blacklist, 
                'whitelist': _precomputed_whitelist,
                'hash': current_hash
            }, f)
    except:
        pass
    
    _db_loaded = True
    print(f"[FaceDB] Finished: {len(_precomputed_blacklist)} blacklist, {len(_precomputed_whitelist)} whitelist")

def _generate_embedding_from_file(path):
    try: 
        # Use 'retinaface' or 'opencv' for initialization as it's more stable for static files
        # then YOLOv8 is used for live video detection.
        results = DeepFace.represent(path, model_name=MODEL_NAME, enforce_detection=False, detector_backend='retinaface')
        if results: return np.array(results[0]['embedding'])
    except Exception as e:
        print(f"[FaceDB Error] Failed to process {os.path.basename(path)}: {e}")
    return None

# ═══════════════════════════════════════════════════════════════
# 2. ANALYSIS & TRACKING
# ═══════════════════════════════════════════════════════════════

def _get_tracker():
    global _tracker
    if _tracker is None:
        _tracker = DeepSort(max_age=30, n_init=2, nms_max_overlap=1.0, max_cosine_distance=0.2)
    return _tracker

def analyze_faces(img_bytes=None, img=None, high_throughput=False):
    """
    Main entry point for face recognition.
    Input: raw bytes or numpy array.
    """
    global _precomputed_blacklist, _precomputed_whitelist
    initialize_databases()
    
    # 1. Decode Image
    if img is None and img_bytes:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None: return {'faces': [], 'total_faces': 0}
    annotated = img.copy()
    
    yolo = get_yolo_model()
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # 2. Detect Faces using YOLOv8
        # We use YOLOv8 to get bounding boxes, then DeepFace for the attributes/embeddings
        yolo_results = yolo(img, verbose=False, device=device, conf=0.4)[0]
        if not yolo_results: return {'faces': [], 'total_faces': 0}
        
        results = []
        for box in yolo_results.boxes:
            # We filter for 'person' (class 0) or 'face' if it's a face-specific model
            # For standard yolov8s.pt, we focus on people and then crop their heads
            if int(box.cls[0]) not in [0]: continue # 0 is person in COCO
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE: continue
            
            # Improve Face Crop: Take upper 40% of the person detection as the face region
            face_y2 = y1 + int((y2 - y1) * 0.4)
            face_y2 = min(y2, max(y1 + 10, face_y2)) # Ensure valid range
            crop = img[y1:face_y2, x1:x2]
            if crop.size == 0: continue
            
            try:
                # Use RetinaFace for more robust attribute analysis within the YOLO crop
                analysis = DeepFace.analyze(crop, actions=['age', 'gender', 'emotion'], 
                                          enforce_detection=True, silent=True, detector_backend='retinaface')
                if isinstance(analysis, list): analysis = analysis[0]
                
                results.append({
                    'region': {'x': x1, 'y': y1, 'w': w, 'h': h}, # Always use original YOLO box for tracking
                    'age': int(analysis.get('age', 0)) if analysis.get('age') else None,
                    'dominant_gender': analysis.get('dominant_gender'),
                    'dominant_emotion': analysis.get('dominant_emotion')
                })
            except Exception as e:
                # Fallback to opencv if retinaface fails on a small crop
                try:
                    analysis = DeepFace.analyze(crop, actions=['age', 'gender', 'emotion'], 
                                              enforce_detection=False, silent=True, detector_backend='opencv')
                    if isinstance(analysis, list): analysis = analysis[0]
                    results.append({
                        'region': {'x': x1, 'y': y1, 'w': w, 'h': h},
                        'age': int(analysis.get('age', 0)) if analysis.get('age') else None,
                        'dominant_gender': analysis.get('dominant_gender'),
                        'dominant_emotion': analysis.get('dominant_emotion')
                    })
                except:
                    results.append({
                        'region': {'x': x1, 'y': y1, 'w': w, 'h': h},
                        'age': None, 'dominant_gender': None, 'dominant_emotion': None
                    })
                
    except Exception as e:
        print(f"[FaceError] YOLO detection failed: {e}")
        return {'faces': [], 'total_faces': 0}

    tracker = _get_tracker()
    detections = []
    
    # 3. Process Detections for Tracker
    for r in results:
        reg = r.get('region', {})
        w, h = reg.get('w', 0), reg.get('h', 0)
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE: continue
        
        # DeepSORT format: [left, top, w, h]
        detections.append(([reg['x'], reg['y'], w, h], 0.9, 'face'))

    # 4. Update Tracker
    tracks = tracker.update_tracks(detections, frame=img)
    face_data_list = []

    # 5. Handle Each Track
    for track in tracks:
        if not track.is_confirmed(): continue
        tid = track.track_id
        
        ltrb = track.to_ltrb() # [left, top, right, bottom]
        x1, y1, x2, y2 = map(int, ltrb)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        
        # Crop & Match (only once per track or periodically)
        name = "Unmatched Person"
        confidence = 0
        status = "unmatched"
        
        cached_emb = _track_embeddings.get(tid)
        if cached_emb is None:
            # Re-crop to focus on the head region (top 40%) for better matching
            face_h = int((y2 - y1) * 0.4)
            face_y2 = min(y2, max(y1 + 10, y1 + face_h))
            face_crop = img[y1:face_y2, x1:x2]
            
            if face_crop.size > 0:
                emb = _generate_embedding(face_crop)
                if emb is not None:
                    _track_embeddings[tid] = emb
                    cached_emb = emb
        
        if cached_emb is not None:
            # Match logic
            best_match = None
            min_dist = 1.0
            
            # Check Blacklist
            for b_name, b_emb in _precomputed_blacklist.items():
                dist = 1 - np.dot(cached_emb, b_emb) # True Cosine Distance
                if dist < min_dist and dist < COSINE_THRESHOLD:
                    min_dist = dist
                    best_match = b_name
                    status = "blacklisted"
            
            # Check Whitelist
            if status == "unmatched":
                for w_name, w_emb in _precomputed_whitelist.items():
                    dist = 1 - np.dot(cached_emb, w_emb)
                    if dist < min_dist and dist < COSINE_THRESHOLD:
                        min_dist = dist
                        best_match = w_name
                        status = "authorized"
            
            if best_match:
                name = best_match
                confidence = round((1 - min_dist) * 100, 1)
                # Ensure status matches Snippet logic
                if status == "authorized":
                    pass # name will be just the best_match
            else:
                name = "Unmatched Person"
                status = "unmatched"

        # 6. Temporal Confirmation
        state = _track_history.get(tid, {"name": "Unmatched Person", "count": 0, "alert_sent": False})
        if status == "blacklisted" and name != "Unmatched Person":
            if state["name"] == name: state["count"] += 1
            else: state = {"name": name, "count": 1, "alert_sent": False}
        else:
            state = {"name": "Unmatched Person", "count": 0, "alert_sent": False}
        _track_history[tid] = state

        is_confirmed = state["count"] >= TEMPORAL_CONFIRM_FRAMES
        
        # 8. Metadata Retrieval (Ensure no nulls for UI)
        age, gender, emotion = "N/A", "Unknown", "Neutral"
        for r in results:
            reg = r['region']
            # Match track bbox with DeepFace attribute region using a more generous threshold
            if abs(reg['x'] - x1) < 150 and abs(reg['y'] - y1) < 150: 
                if r.get('age') is not None: age = int(r['age'])
                if r.get('dominant_gender'): gender = str(r['dominant_gender'])
                if r.get('dominant_emotion'): emotion = str(r['dominant_emotion'])
                break

        # 8. Alerts & Evidence
        if is_confirmed and status == "blacklisted" and not state.get("alert_sent"):
            # Trigger both local log/evidence and email alert
            _trigger_alert(name, confidence, img)
            send_alert_email(
                subject=f"BLACKLIST ALERT: {name} Spotted",
                message=f"System identified blacklisted individual: {name} in the monitored area.",
                alert_type="face"
            )
            state["alert_sent"] = True

        # 9. Drawing Premium UI
        color = (0, 255, 255) # Yellow
        if status == "blacklisted": color = (0, 140, 255) if not is_confirmed else (0, 0, 255) # Orange -> Red
        if status == "authorized": color = (0, 255, 0) # Green
        
        # Draw Main Bounding Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Prepare Label Content based on user snippet logic
        if status == "blacklisted":
            display_name = f"Blacklisted: {name}" if is_confirmed else f"Verifying: {name}"
            color = (0, 0, 255) if is_confirmed else (0, 140, 255) # Red for confirmed, Orange for verifying
        elif status == "authorized":
            display_name = name
            color = (0, 255, 0) # Green
        else: # status == "unmatched"
            display_name = "Unmatched Person"
            color = (0, 255, 255) # Yellow

        info_text = f"{display_name} ({confidence}%)" if confidence > 0 else display_name
        attr_text = f"{gender}, {age}"
        
        # Draw Label Background Box
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thick = 1
        (w1, h1), _ = cv2.getTextSize(info_text, font, scale, thick)
        (w2, h2), _ = cv2.getTextSize(attr_text, font, scale - 0.1, thick)
        max_w = max(w1, w2) + 10
        
        # Label above or inside?
        label_y = y1 if y1 > 40 else y1 + h1 + h2 + 20
        cv2.rectangle(annotated, (x1, label_y - h1 - h2 - 15), (x1 + max_w, label_y), color, -1)
        
        # Text
        cv2.putText(annotated, info_text, (x1 + 5, label_y - h2 - 10), font, scale, (0, 0, 0), 2)
        cv2.putText(annotated, attr_text, (x1 + 5, label_y - 5), font, scale - 0.1, (50, 50, 50), 1)
        
        # 10. Prepare API response data

        face_data_list.append({
            'name': name,
            'confidence': confidence,
            'status': status,
            'is_blacklisted': status == "blacklisted",
            'is_authorized': status == "authorized",
            'confirmed': is_confirmed,
            'id': str(tid), # Ensure ID is always a string for frontend consistency
            'age': age,
            'gender': gender,
            'emotion': emotion,
            'bbox': [x1, y1, x2, y2]
        })

    # Encode Result
    _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {
        'faces': face_data_list,
        'total_faces': len(face_data_list),
        'annotated_image': base64.b64encode(buffer).decode('utf-8')
    }

def _trigger_alert(name, conf, frame):
    """Log alert and save evidence."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    unix_ts = int(time.time())
    
    # Save Image
    fname = f"match_{name}_{unix_ts}.jpg"
    fpath = os.path.join(ALERTS_DIR, fname)
    cv2.imwrite(fpath, frame)
    
    # Log text
    msg = f"[{ts}] BLACKLIST HIT: {name} (Conf: {conf}%) - IP: Local - Evidence: {fname}\n"
    with open(ALERTS_LOG, 'a') as f:
        f.write(msg)
    print(f"!!! ALERT !!! {name} matched with {conf}% confidence")

if __name__ == "__main__":
    # Self-test
    initialize_databases(force_refresh=True)
