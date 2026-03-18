import os
import cv2
import numpy as np
import pickle
import time
from ultralytics import YOLO
from deepface import DeepFace
from security_utils import send_alert_email

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
BLACKLIST_DIR = os.path.join(BASE_DIR, 'face_blacklist_db')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
CACHE_PATH = os.path.join(BASE_DIR, 'robust_blacklist_cache.pkl')

_blacklist_embeddings = {}
_db_loaded = False
_yolo_model = None
_track_history = {}  # {track_id: {name: count}}
_alerted_tracks = set()

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        model_path = os.path.join(MODELS_DIR, 'yolov8n.pt')
        _yolo_model = YOLO(model_path)
    return _yolo_model

def build_blacklist_db():
    global _blacklist_embeddings, _db_loaded
    if _db_loaded:
        return
    
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, 'rb') as f:
                _blacklist_embeddings = pickle.load(f)
                _db_loaded = True
                print(f"[RobustBlacklist] Loaded {len(_blacklist_embeddings)} known blacklist identities from cache.")
                return
        except Exception as e:
            print(f"[RobustBlacklist] Cache load failed: {e}")
            
    print("[RobustBlacklist] Building face embeddings database...")
    if not os.path.exists(BLACKLIST_DIR):
        os.makedirs(BLACKLIST_DIR)
        
    for entry in os.listdir(BLACKLIST_DIR):
        person_dir = os.path.join(BLACKLIST_DIR, entry)
        if not os.path.isdir(person_dir): 
            continue
        
        embs = []
        for img_name in os.listdir(person_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): 
                continue
            
            img_path = os.path.join(person_dir, img_name)
            try:
                # Use RetinaFace for robust face detection and FaceNet for embeddings
                res = DeepFace.represent(img_path, model_name="Facenet", detector_backend="retinaface", enforce_detection=False)
                if res and len(res) > 0:
                    emb = np.array(res[0]['embedding'])
                    emb = emb / np.linalg.norm(emb)
                    embs.append(emb)
            except Exception as e:
                print(f"[RobustBlacklist] Failed to process {img_path}: {e}")
                
        if len(embs) > 0:
            avg_emb = np.mean(embs, axis=0)
            avg_emb = avg_emb / np.linalg.norm(avg_emb)
            _blacklist_embeddings[entry] = avg_emb
            print(f"[RobustBlacklist] Added {entry} with {len(embs)} encodings.")
            
    try:
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump(_blacklist_embeddings, f)
    except Exception as e:
        print(f"[RobustBlacklist] Failed to cache: {e}")
        
    _db_loaded = True

def detect_blacklist(image_bytes):
    """
    Detects faces matching the blacklist from the provided image bytes.
    Uses YOLOv8 for person detection, DeepFace for face extraction and comparison.
    """
    build_blacklist_db()
    
    if len(_blacklist_embeddings) == 0:
        return {'matches': []}
        
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {'matches': []}
        
    yolo = get_yolo_model()
    # Track to keep history for temporal confirmation (persist=True)
    results = yolo.track(img, persist=True, classes=[0], verbose=False) # 0 is person
    
    if not results or not results[0].boxes or results[0].boxes.id is None:
        return {'matches': []}
        
    boxes = results[0].boxes.xyxy.cpu().numpy()
    ids = results[0].boxes.id.cpu().numpy()
    
    matches = []
    current_ids = set()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    for i in range(len(boxes)):
        track_id = int(ids[i])
        current_ids.add(track_id)
        
        x1, y1, x2, y2 = map(int, boxes[i])
        w, h = x2 - x1, y2 - y1
        if w < 50 or h < 50:
            continue
            
        crop = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
        if crop.size == 0:
            continue
            
        # Try to extract embedding from the person crop
        try:
            # We use RetinaFace to detect and extract the face inside the person crop
            res = DeepFace.represent(crop, model_name="Facenet", detector_backend="retinaface", enforce_detection=False)
            if not res:
                continue
                
            face_emb = np.array(res[0]['embedding'])
            face_emb = face_emb / np.linalg.norm(face_emb)
            
            best_match = None
            best_sim = -1
            
            # Compare with blacklist
            for name, b_emb in _blacklist_embeddings.items():
                sim = np.dot(face_emb, b_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_match = name
                    
            if best_match and best_sim > 0.60: # Threshold for Facenet cosine similarity
                if track_id not in _track_history:
                    _track_history[track_id] = {}
                if best_match not in _track_history[track_id]:
                    _track_history[track_id][best_match] = 0
                    
                _track_history[track_id][best_match] += 1
                
                # Requires 2 consecutive frames for temporal confirmation to reduce false positives
                if _track_history[track_id][best_match] >= 2:
                    matches.append({
                        "name": best_match,
                        "confidence": float(best_sim),
                        "bbox": [x1, y1, x2, y2],
                        "track_id": track_id,
                        "timestamp": timestamp,
                        "camera_id": "CAM_MAIN"
                    })
                    
                    if track_id not in _alerted_tracks:
                        _alerted_tracks.add(track_id)
                        # Trigger alert
                        send_alert_email(
                             subject=f"CRITICAL: BLACKLIST MATCH ({best_match})",
                             message=f"Blacklisted individual '{best_match}' detected!\n"
                                     f"Confidence: {best_sim*100:.2f}%\n"
                                     f"Camera: CAM_MAIN\n"
                                     f"Time: {timestamp}",
                             alert_type="blacklist_match"
                        )
        except Exception:
            pass
            
    # Cleanup old tracks to free memory
    stale = [tid for tid in _track_history.keys() if tid not in current_ids]
    for tid in stale:
        del _track_history[tid]
        if tid in _alerted_tracks:
            _alerted_tracks.remove(tid)
            
    return {'matches': matches}
