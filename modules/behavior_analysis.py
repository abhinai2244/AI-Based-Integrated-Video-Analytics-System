"""
Behavior Analysis Module v2.0
==============================
Uses YOLOv8-Pose to detect human keypoints and analyze specific behaviors:
1. FALL DETECTION    — Detects person lying on ground (fainted/collapsed)
2. FIGHT DETECTION   — Detects two people in close proximity with aggressive poses
3. LOITERING         — Detects person staying in area too long
4. AGGRESSION        — Rapid limb movement / raised arms in fighting stance

Detection Logic:
  Fall:   Aspect ratio (wide > tall), torso vertical angle, hip-shoulder Y overlap
  Fight:  Two+ people overlapping bounding boxes + raised/extended arms + velocity spikes
  Loiter: Person present for > LOITERING_THRESHOLD seconds
"""

import cv2
import numpy as np
from ultralytics import YOLO
import base64
import os
import time
import math
from security_utils import send_alert_email

# Models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
YOLO_POSE_MODEL = os.path.join(MODELS_DIR, 'yolov8n-pose.pt')

_model = None

# ── Keypoint indices (COCO 17-point format) ──────────────────────────────
# 0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear
# 5:left_shoulder, 6:right_shoulder, 7:left_elbow, 8:right_elbow
# 9:left_wrist, 10:right_wrist, 11:left_hip, 12:right_hip
# 13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle

SKELETON = [
    (0,1),(0,2),(1,3),(2,4),            # Head
    (5,6),(5,7),(7,9),(6,8),(8,10),     # Arms
    (5,11),(6,12),(11,12),              # Torso
    (11,13),(13,15),(12,14),(14,16)     # Legs
]

# Global per-track history
# { track_id: {first_seen, last_seen, positions: [(cx,cy)], pose_history: [keypoints]} }
_behavior_history = {}

LOITERING_THRESHOLD = 30   # seconds before loitering alert
FIGHT_OVERLAP_THRESH = 0.12  # Lowered IoU threshold for proximity-based fight detection
FIGHT_SCORE_THRESHOLD = 0.42 # Lowered from 0.50 for faster triggers
CONF_THRESHOLD = 0.35         # Lowered from 0.4 to keep more keypoints

# Maximum centroid distance (pixels) to consider same person when track ID changes
# Set relative to image size — resolved at runtime
MAX_CENTROID_MATCH_DIST = 120  # pixels


def get_model():
    global _model
    if _model is None:
        model_path = YOLO_POSE_MODEL if os.path.exists(YOLO_POSE_MODEL) else 'yolov8n-pose.pt'
        _model = YOLO(model_path)
        print(f"[Behavior] Loaded pose model: {model_path}")
    return _model


def _centroid(box):
    """Return centre of a (x1,y1,x2,y2) box."""
    return ((box[0]+box[2])/2.0, (box[1]+box[3])/2.0)


def _resolve_track_id(new_tid, box, current_time):
    """
    Find the best-matching EXISTING history entry for this bounding box by
    centroid proximity. If a match is found within MAX_CENTROID_MATCH_DIST,
    return that entry's key (preserving the loitering clock).
    If no match, return new_tid so a fresh entry is created.

    This prevents 'loitering resets to Normal' when the YOLO tracker
    assigns a new track ID to a person that was already being tracked.
    """
    cx, cy = _centroid(box)
    best_old_tid = None
    best_dist = MAX_CENTROID_MATCH_DIST

    for old_tid, hist in _behavior_history.items():
        if old_tid == new_tid:
            return new_tid   # already same ID, no remapping needed
        # Only consider entries seen very recently (within 3 seconds)
        if current_time - hist['last_seen'] > 3.0:
            continue
        ox, oy = hist.get('last_centroid', (cx+9999, cy+9999))
        d = math.hypot(cx - ox, cy - oy)
        if d < best_dist:
            best_dist = d
            best_old_tid = old_tid

    return best_old_tid if best_old_tid is not None else new_tid


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _iou(boxA, boxB):
    """Intersection over Union of two (x1,y1,x2,y2) boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(1, (boxA[2]-boxA[0]) * (boxA[3]-boxA[1]))
    boxBArea = max(1, (boxB[2]-boxB[0]) * (boxB[3]-boxB[1]))
    return interArea / float(boxAArea + boxBArea - interArea)


def _pt(kps, idx):
    """Return (x, y) if keypoint confidence is above threshold, else None."""
    if kps[idx][2] >= CONF_THRESHOLD:
        return (float(kps[idx][0]), float(kps[idx][1]))
    return None


def _angle(p1, p2, p3):
    """Angle at p2 formed by p1-p2-p3 (in degrees)."""
    if None in (p1, p2, p3):
        return None
    v1 = (p1[0]-p2[0], p1[1]-p2[1])
    v2 = (p3[0]-p2[0], p3[1]-p2[1])
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag = math.hypot(*v1) * math.hypot(*v2)
    if mag < 1e-6: return None
    return math.degrees(math.acos(max(-1, min(1, dot/mag))))


def _dist(p1, p2):
    if p1 is None or p2 is None: return 9999
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])


# ─────────────────────────────────────────────────────────────────────────────
# --- FALL DETECTION ---
# ─────────────────────────────────────────────────────────────────────────────

def _detect_fall(kps, box, img_h):
    """
    Multi-signal fall detection:
    1. BBox aspect ratio: width > height suggests horizontal posture
    2. Torso angle: shoulder-hip vector far from vertical
    3. Head-to-hip vertical position: head Y is close to hip Y (lying flat)
    4. Hip height: hips closer to bottom of image (near ground)
    """
    x1, y1, x2, y2 = box
    width  = x2 - x1
    height = y2 - y1
    if height == 0: return False, 0.0

    aspect_ratio = width / height  # > 1.3 means wider than tall → horizontal
    score = 0.0

    # Signal 1: Aspect ratio
    if aspect_ratio > 1.5:
        score += 0.45
    elif aspect_ratio > 1.1:
        score += 0.2

    # Signal 2: Torso angle (shoulder → hip should be ~vertical when standing)
    l_shoulder = _pt(kps, 5)
    r_shoulder = _pt(kps, 6)
    l_hip      = _pt(kps, 11)
    r_hip      = _pt(kps, 12)

    if l_shoulder and l_hip:
        dx = l_hip[0] - l_shoulder[0]
        dy = l_hip[1] - l_shoulder[1]
        torso_angle = abs(math.degrees(math.atan2(dx, dy + 1e-6)))  # 0=vertical
        if torso_angle > 60:   # Torso is mostly horizontal
            score += 0.35
        elif torso_angle > 35:
            score += 0.15

    # Signal 3: Head Y close to hip Y (both at same vertical level → lying)
    nose = _pt(kps, 0)
    if nose and l_hip and r_hip:
        mid_hip_y = (l_hip[1] + r_hip[1]) / 2.0
        head_hip_diff = abs(nose[1] - mid_hip_y) / (height + 1e-6)
        if head_hip_diff < 0.25:  # Head and hips at nearly same height
            score += 0.3
        elif head_hip_diff < 0.45:
            score += 0.1

    # Signal 4: Position in image (person lying close to bottom half)
    # (Not definitive alone, but combined with others)
    box_center_y = (y1 + y2) / 2.0
    if box_center_y > img_h * 0.5 and aspect_ratio > 0.9:
        score += 0.1

    return score >= 0.55, round(score, 2)


# ─────────────────────────────────────────────────────────────────────────────
# --- FIGHT DETECTION ---
# ─────────────────────────────────────────────────────────────────────────────

def _detect_fight_pair(kpsA, boxA, kpsB, boxB):
    """
    Enhanced fight detection between two people:
    - BBox overlap & proximity
    - Arm/wrist positions relative to other person (hitting/pushing stance)
    - Relative movements (checked via combined score)
    """
    iou = _iou(boxA, boxB)
    score = 0.0

    # Signal 1: Proximity / overlap (IoU)
    if iou > 0.15: # Lowered from 0.18
        score += 0.50 # Increased from 0.45
    elif iou > 0.05:
        score += 0.35 # Increased from 0.25
        
    # Signal 2: Centroid Proximity (Even if boxes don't overlap much)
    cenA = _centroid(boxA)
    cenB = _centroid(boxB)
    dist = _dist(cenA, cenB)
    avg_h = ((boxA[3]-boxA[1]) + (boxB[3]-boxB[1])) / 2.0
    if dist < avg_h * 0.85: # Increased from 0.7 for better sensitivity
        score += 0.25 # Increased from 0.2

    # Signal 4: Punch/Push Detection (Elbow extension toward other person)
    def is_extending_toward(kps_self, box_other):
        l_s = _pt(kps_self, 5); l_e = _pt(kps_self, 7); l_w = _pt(kps_self, 9)
        r_s = _pt(kps_self, 6); r_e = _pt(kps_self, 8); r_w = _pt(kps_self, 10)
        
        ox1, oy1, ox2, oy2 = box_other
        extending = 0
        for s, e, w in [(l_s, l_e, l_w), (r_s, r_e, r_w)]:
            if s and e and w:
                ang = _angle(s, e, w)
                if ang and ang > 125: # Even more lenient for high-intensity movement
                    # Check if wrist is closer to other person than shoulder
                    dist_w = _dist(w, cenB)
                    dist_s = _dist(s, cenB)
                    if dist_w < dist_s:
                        extending += 1
        return extending

    extA = is_extending_toward(kpsA, boxB)
    extB = is_extending_toward(kpsB, boxA)
    if extA + extB >= 1:
        score += 0.25
    
    # Signal 5: Raised Arms (Classic fighting stance)
    def arms_up(kps):
        s = 0
        l_w = _pt(kps, 9); l_s = _pt(kps, 5); l_e = _pt(kps, 7)
        r_w = _pt(kps, 10); r_s = _pt(kps, 6); r_e = _pt(kps, 8)
        # Wrist above shoulder level or Elbow high
        if l_w and l_s and l_w[1] < l_s[1]: s += 1
        elif l_e and l_s and l_e[1] < l_s[1]: s += 0.5
        
        if r_w and r_s and r_w[1] < r_s[1]: s += 1
        elif r_e and r_s and r_e[1] < r_s[1]: s += 0.5
        return s

    upA = arms_up(kpsA)
    upB = arms_up(kpsB)
    if upA + upB >= 1.5:
        score += 0.25

    # Vertical consistency: people should be on a similar ground level
    # (prevents matching someone in background with someone in foreground)
    groundA = boxA[3]
    groundB = boxB[3]
    if abs(groundA - groundB) > avg_h * 0.8:
        score *= 0.5 # Penalty for depth mismatch

    return score >= FIGHT_SCORE_THRESHOLD, round(score, 2)


# ─────────────────────────────────────────────────────────────────────────────
# --- VELOCITY / ARM SPEED ---
# ─────────────────────────────────────────────────────────────────────────────

def _calc_wrist_velocity(tid, kps, current_time):
    """Returns average wrist speed based on history."""
    hist = _behavior_history.get(tid, {})
    prev_poses = hist.get('pose_history', [])

    l_wrist = _pt(kps, 9)
    r_wrist = _pt(kps, 10)

    if len(prev_poses) >= 1:
        prev_kps, prev_t = prev_poses[-1]
        dt = max(current_time - prev_t, 0.05)
        speeds = []
        prev_lw = _pt(prev_kps, 9)
        prev_rw = _pt(prev_kps, 10)
        if l_wrist and prev_lw:
            speeds.append(_dist(l_wrist, prev_lw) / dt)
        if r_wrist and prev_rw:
            speeds.append(_dist(r_wrist, prev_rw) / dt)
        return np.mean(speeds) if speeds else 0.0
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def detect_behaviors(image_bytes=None, img=None, high_throughput=False):
    if img is None:
        if image_bytes is None:
            return {"error": "No image data provided"}
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Could not decode image"}

    model = get_model()
    current_time = time.time()
    img_h, img_w = img.shape[:2]

    # ── Run YOLOv8-Pose tracking ─────────────────────────────────────────────
    try:
        results = model.track(img, persist=True, verbose=False, conf=0.3)[0]
    except Exception as e:
        print(f"[Behavior] Tracking error: {e}")
        try:
            results = model(img, verbose=False, conf=0.3)[0]
        except:
            return {"error": str(e)}

    annotated = img.copy()
    detections = []
    alerts = []

    # ── Collect all detected people ──────────────────────────────────────────
    people = []  # list of {tid, box, kps}

    has_ids = results.boxes.id is not None
    boxes_arr = results.boxes.xyxy.cpu().numpy()
    ids_arr   = results.boxes.id.cpu().numpy().astype(int) if has_ids else \
                np.arange(len(boxes_arr))

    if results.keypoints is not None and len(results.keypoints.data) > 0:
        kps_arr = results.keypoints.data.cpu().numpy()
    else:
        # No keypoints available — fall back to boxes-only heuristic
        kps_arr = None

    for i in range(len(boxes_arr)):
        box = boxes_arr[i]
        raw_tid = int(ids_arr[i])
        kps = kps_arr[i] if kps_arr is not None else None

        # ── Centroid-based identity preservation ──────────────────────────────
        # If YOLO re-assigned a new track ID to the same physical person,
        # remap to the old ID so the loitering clock is not reset.
        resolved_tid = _resolve_track_id(raw_tid, box, current_time)
        tid = resolved_tid

        people.append({'tid': tid, 'box': box, 'kps': kps})

        # Initialise / update history
        cx, cy = _centroid(box)
        if tid not in _behavior_history:
            _behavior_history[tid] = {
                'first_seen':    current_time,
                'last_seen':     current_time,
                'last_centroid': (cx, cy),
                'pose_history':  [],
                'alerted_fight': False,
                'alerted_fall':  False,
                'fight_confirm_count': 0,
                'persistent_behavior': 'Normal',
                'behavior_sticky_until': 0,
            }
        else:
            _behavior_history[tid]['last_seen']     = current_time
            _behavior_history[tid]['last_centroid'] = (cx, cy)

        if kps is not None:
            hist = _behavior_history[tid]
            hist['pose_history'].append((kps, current_time))
            if len(hist['pose_history']) > 10:
                hist['pose_history'] = hist['pose_history'][-10:]

    # ── Analyse each person ──────────────────────────────────────────────────
    person_behaviors = {}   # tid -> behavior label

    for p in people:
        tid = p['tid']
        box = p['box']
        kps = p['kps']
        x1, y1, x2, y2 = map(int, box)
        hist = _behavior_history[tid]
        duration = current_time - hist['first_seen']

        behavior = "Normal"
        confidence_score = 0.0
        color = (0, 220, 100)  # Green

        # ── 1. FALL DETECTION ────────────────────────────────────────────────
        fall_detected = False
        fall_score = 0.0
        if kps is not None:
            fall_detected, fall_score = _detect_fall(kps, (x1, y1, x2, y2), img_h)
        else:
            # Box-only fallback
            w = x2 - x1; h = max(y2 - y1, 1)
            fall_detected = (w / h) > 1.6
            fall_score = 0.6 if fall_detected else 0.0

        # ── 3. PERSISTENCE CHECK ─────────────────────────────────────────────
        # If we have a high-priority sticky behavior, keep it
        if current_time < hist.get('behavior_sticky_until', 0):
            if hist['persistent_behavior'] in ["FALL DETECTED", "FIGHTING"]:
                behavior = hist['persistent_behavior']
                confidence_score = 0.9 # High confidence for persistent state
        
        # Priority: Fall > Fight > Loitering (Fight added later)
        if fall_detected:
            behavior = "FALL DETECTED"
            hist['persistent_behavior'] = "FALL DETECTED"
            hist['behavior_sticky_until'] = current_time + 4.0 # Stick for 4s
            color = (0, 0, 255)
            confidence_score = fall_score
            alerts.append(f"Person {tid} has fallen / fainted!")
            if not hist['alerted_fall']:
                hist['alerted_fall'] = True
                send_alert_email("MEDICAL EMERGENCY: Person Fallen", f"Person ID {tid} lying on ground.", "behavior_fall")
        
        # ── 2. LOITERING ────────────────────────────────────────────────────
        if behavior == "Normal" and duration > LOITERING_THRESHOLD:
            behavior = "LOITERING"
            # No sticky for loitering, it should be real-time
            color = (0, 165, 255)
            confidence_score = 0.6
        
        person_behaviors[tid] = behavior

        # Store detection (fight added below after pair analysis)
        detections.append({
            "id": int(tid),
            "behavior": behavior,
            "bbox": [x1, y1, x2, y2],
            "duration": round(duration, 1),
            "confidence": confidence_score
        })

    # ── 3. FIGHT & AGGRESSION DETECTION (pair-wise) ───────────────────────────
    fight_pairs = set()
    if len(people) >= 2:
        for i in range(len(people)):
            for j in range(i+1, len(people)):
                pA, pB = people[i], people[j]
                fight_flag, fight_score = False, 0.0
                
                if pA['kps'] is None or pB['kps'] is None:
                    iou = _iou(pA['box'], pB['box'])
                    fight_flag = iou > 0.15
                    fight_score = iou
                else:
                    fight_flag, fight_score = _detect_fight_pair(
                        pA['kps'], pA['box'],
                        pB['kps'], pB['box']
                    )
                    # Boost if wrists are moving fast
                    vel_A = _calc_wrist_velocity(pA['tid'], pA['kps'], current_time)
                    vel_B = _calc_wrist_velocity(pB['tid'], pB['kps'], current_time)
                    if vel_A > 60 or vel_B > 60:  # More sensitive threshold (pixels/sec)
                        fight_score = min(1.0, fight_score + 0.3)
                        if fight_score >= 0.40:
                            fight_flag = True

                if fight_flag:
                    # Update confirmation counters
                    for p in [pA, pB]:
                        hist = _behavior_history.get(p['tid'], {})
                        hist['fight_confirm_count'] = hist.get('fight_confirm_count', 0) + 1

                    # If score is very high (> 0.6), trigger instantly, otherwise wait for 1-frame confirm
                    hist_A = _behavior_history.get(pA['tid'], {})
                    if fight_score > 0.60 or hist_A.get('fight_confirm_count', 0) >= 2:
                        fight_pairs.add((pA['tid'], pB['tid']))
                        
                        # Send alert once per pair
                        if not _behavior_history[pA['tid']].get('alerted_fight'):
                            _behavior_history[pA['tid']]['alerted_fight'] = True
                            send_alert_email(
                                "SECURITY ALERT: Physical Conflict Detected",
                                f"Persons {pA['tid']} and {pB['tid']} in active conflict. (Intensity: {fight_score:.2f})",
                                "behavior_fight"
                            )
                else:
                    # Reset counters if no longer fighting
                    for p in [pA, pB]:
                        hist = _behavior_history.get(p['tid'], {})
                        if 'fight_confirm_count' in hist:
                            hist['fight_confirm_count'] = max(0, hist['fight_confirm_count'] - 1)

    # Update detections list with high-priority fighting status
    for d in detections:
        tid = d['id']
        hist = _behavior_history.get(tid, {})
        is_in_fight = any(tid == pair[0] or tid == pair[1] for pair in fight_pairs)
        
        # Priority: Fall already set. Now apply Fight.
        if d['behavior'] == "FALL DETECTED":
            continue
            
        if is_in_fight:
            d['behavior'] = "FIGHTING"
            d['confidence'] = 0.95
            hist['persistent_behavior'] = "FIGHTING"
            hist['behavior_sticky_until'] = current_time + 4.0 # Stick for 4s
            if f"Person {tid} is involved in a fight!" not in alerts:
                alerts.append(f"Person {tid} is involved in a fight!")
        elif current_time < hist.get('behavior_sticky_until', 0) and hist['persistent_behavior'] == "FIGHTING":
            # Persistence kick-in
            d['behavior'] = "FIGHTING"
            d['confidence'] = 0.8 # Slightly lower confidence for persistent state
        elif d['behavior'] == "Normal":
            # Check for aggression
            p_data = next((p for p in people if p['tid'] == tid), None)
            if p_data and p_data['kps'] is not None:
                vel = _calc_wrist_velocity(tid, p_data['kps'], current_time)
                if vel > 120:
                    d['behavior'] = "AGGRESSIVE"
                    d['confidence'] = 0.6

    # ── Draw Annotations ──────────────────────────────────────────────────────
    BEHAVIOR_COLORS = {
        "Normal":        (0, 220, 100),
        "FALL DETECTED": (0, 0, 255),
        "FIGHTING":      (0, 80, 255),
        "LOITERING":     (0, 165, 255),
        "AGGRESSIVE":    (255, 0, 165), # Purple/Magenta
    }
    SKELETON_COLORS = {
        "Normal":        (180, 255, 180),
        "FALL DETECTED": (100, 100, 255),
        "FIGHTING":      (80, 140, 255),
        "LOITERING":     (100, 200, 255),
        "AGGRESSIVE":    (220, 150, 255),
    }

    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        tid = d['id']
        behavior = d['behavior']
        color = BEHAVIOR_COLORS.get(behavior, (0,220,100))
        skel_color = SKELETON_COLORS.get(behavior, (180,255,180))

        # Bounding box + label
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"#{tid} {behavior}"
        conf_text = f"{d['confidence']*100:.0f}%" if d['confidence'] > 0 else ""
        text_y = max(y1 - 10, 14)
        cv2.putText(annotated, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        if conf_text:
            cv2.putText(annotated, conf_text, (x1, text_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw skeleton
        person_data = next((p for p in people if p['tid'] == tid), None)
        if person_data and person_data['kps'] is not None:
            kps = person_data['kps']
            for start_idx, end_idx in SKELETON:
                if kps[start_idx][2] > 0.4 and kps[end_idx][2] > 0.4:
                    p1 = (int(kps[start_idx][0]), int(kps[start_idx][1]))
                    p2 = (int(kps[end_idx][0]), int(kps[end_idx][1]))
                    cv2.line(annotated, p1, p2, skel_color, 2, cv2.LINE_AA)
            # Draw keypoints
            for k_idx in range(17):
                if kps[k_idx][2] > 0.5:
                    cx, cy = int(kps[k_idx][0]), int(kps[k_idx][1])
                    cv2.circle(annotated, (cx, cy), 4, color, -1, cv2.LINE_AA)

    # ── HUD status banner ─────────────────────────────────────────────────────
    alert_behaviors = [d['behavior'] for d in detections if d['behavior'] != 'Normal']
    if alert_behaviors:
        unique_alerts = list(set(alert_behaviors))
        status_text = "⚠ ALERT: " + " | ".join(unique_alerts)
        banner_color = (30, 0, 200) if "FIGHTING" in unique_alerts else (0, 0, 200)
        cv2.rectangle(annotated, (0, 0), (img_w, 35), banner_color, -1)
        cv2.putText(annotated, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(annotated, (0, 0), (img_w, 35), (0, 120, 0), -1)
        cv2.putText(annotated, f"Status: Normal — {len(detections)} person(s) tracked", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 2, cv2.LINE_AA)

    # ── Cleanup stale tracks ──────────────────────────────────────────────────
    stale = [k for k, v in _behavior_history.items() if current_time - v['last_seen'] > 15]
    for k in stale:
        del _behavior_history[k]

    # ── Encode result ─────────────────────────────────────────────────────────
    _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    annotated_b64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "status": "alert" if alerts else "clear",
        "alerts": alerts,
        "detections": detections,
        "annotated_image": annotated_b64,
        "total_active": len(detections),
        "fight_detected": len(fight_pairs) > 0,
        "fall_detected": any(d['behavior'] == 'FALL DETECTED' for d in detections),
    }
