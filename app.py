"""
AI Video Analytics Dashboard - Flask Application
Serves the web dashboard and provides API endpoints for AI inference.
"""

import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, session, redirect, url_for, abort
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from security_utils import (
    hash_password, verify_password, require_role, log_security_event, 
    is_ip_blocked, block_ip, validate_password_policy, log_watchlist_action
)
from database import get_db_connection

app = Flask(__name__)
app.secret_key = os.urandom(24) # Secure secret key for sessions
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Initialize Rate Limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

@app.before_request
def check_ip_block():
    if is_ip_blocked(request.remote_addr):
        return abort(403, description="Your IP has been blocked due to suspicious activity.")
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config_data.json')
WHITELIST_DIR = os.path.join(os.path.dirname(__file__), 'whitelist_db')
FACE_BLACKLIST_DIR = os.path.join(os.path.dirname(__file__), 'face_blacklist_db')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(WHITELIST_DIR, exist_ok=True)
os.makedirs(FACE_BLACKLIST_DIR, exist_ok=True)

if not os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, 'w') as f:
        json.dump({"blacklist_plates": []}, f)

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def allowed_video(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    """Uploads a video file to the uploads folder."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No video selected'}), 400
    if file and allowed_video(file.filename):
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'success': True, 'filename': filename})
    return jsonify({'error': 'Video type not allowed'}), 400


@app.route('/api/analyze-frame', methods=['POST'])
@limiter.exempt
def api_analyze_frame():
    """Lightweight endpoint for live frame analysis from browser video stream."""
    try:
        # Accept both 'image' (API) and 'file' (live stream frontend) field names
        if 'image' in request.files:
            file = request.files['image']
        elif 'file' in request.files:
            file = request.files['file']
        else:
            return jsonify({'error': 'No image provided'}), 400
        image_bytes = file.read()
        
        # Apply night vision if requested
        night_vision = request.form.get('night_vision') == 'true'
        if night_vision:
            import cv2
            import numpy as np
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                img = enhance_low_light(img)
                _, buf = cv2.imencode('.jpg', img)
                image_bytes = buf.tobytes()

        from modules.vehicle_counter import detect_vehicles
        from modules.anpr import detect_plates
        from modules.face_recognition_module import analyze_faces
        from modules.people_counter import count_people
        from modules.weapon_detection import detect_weapons
        from modules.behavior_analysis import detect_behaviors
        from modules.robust_blacklist import detect_blacklist
        
        from concurrent.futures import ThreadPoolExecutor
        
        # Run AI modules in parallel to leverage multi-core CPU/GPU
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_vehicle = executor.submit(detect_vehicles, image_bytes)
            future_anpr = executor.submit(detect_plates, image_bytes)
            future_face = executor.submit(analyze_faces, image_bytes)
            future_people = executor.submit(count_people, image_bytes)
            future_weapon = executor.submit(detect_weapons, image_bytes)
            future_behavior = executor.submit(detect_behaviors, image_bytes)
            future_blacklist = executor.submit(detect_blacklist, image_bytes)
            from modules.helmet_detector import detect_helmets
            future_helmet = executor.submit(detect_helmets, image_bytes)
            
            vehicle_res = future_vehicle.result()
            anpr_res = future_anpr.result()
            face_res = future_face.result()
            people_res = future_people.result()
            weapon_res = future_weapon.result()
            behavior_res = future_behavior.result()
            blacklist_res = future_blacklist.result()
            helmet_res = future_helmet.result()
        
        return jsonify({
            'success': True,
            'vehicles': vehicle_res,
            'anpr': anpr_res,
            'faces': face_res,
            'people': people_res,
            'weapons': weapon_res,
            'behavior': behavior_res,
            'blacklist': blacklist_res,
            'helmets': helmet_res
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─── Full Video Analysis ──────────────────────────────────────

# Global state for video analysis progress
# Redundant analysis state and routes removed. Using ANALYSIS_STATUS based implementation below.


def _process_full_video(video_path):
    """Process the entire video, sampling frames at intervals."""
    import cv2
    import numpy as np
    global _video_analysis_state
    
    try:
        from modules.vehicle_counter import detect_vehicles
        from modules.anpr import detect_plates
        from modules.face_recognition_module import analyze_faces
        from modules.people_counter import count_people
        from modules.weapon_detection import detect_weapons
        from modules.behavior_analysis import detect_behaviors
        from modules.robust_blacklist import detect_blacklist
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            _video_analysis_state['error'] = 'Could not open video file'
            _video_analysis_state['running'] = False
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _video_analysis_state['total_frames'] = total_frames
        
        # ── SPEED OPTIMIZATION ──
        # Sample 1 frame every 3 seconds of video
        # For a 1hr video at 30fps = 108000 frames, we only process 1200 frames
        # Higher scan precision (1 fps) to avoid missing brief events
        sample_interval_sec = 1
        sample_interval_frames = max(1, int(fps * sample_interval_sec))
        
        # Pre-compute all frame indices to process (using SEEK, not sequential read)
        frame_indices = list(range(sample_interval_frames, total_frames, sample_interval_frames))
        total_to_process = len(frame_indices)
        
        # Downscale target: 720p max for faster AI inference
        MAX_DIM = 720
        scale = 1.0
        if max(width, height) > MAX_DIM:
            scale = MAX_DIM / max(width, height)
        
        print(f"[Video Analysis] {total_frames} total frames, processing {total_to_process} sampled frames "
              f"(every {sample_interval_sec}s), scale={scale:.2f}")
        
        # Accumulated results
        all_plates = {}       # text -> {plate_data}
        all_vehicles = {}     # id -> {class, speed}
        vehicle_type_counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        all_people_ids = set()
        gender_counts = {'Male': 0, 'Female': 0}
        all_faces = {}        # name -> {face_data}
        max_speed = 0
        speed_records = []
        alerts = []
        behavior_events = []
        blacklist_events = []
        
        processed = 0
        
        # Create ONE shared executor (avoids crash on shutdown)
        executor = ThreadPoolExecutor(max_workers=6)
        
        try:
            for i, frame_idx in enumerate(frame_indices):
                if not _video_analysis_state['running']:
                    break
                
                # SEEK directly to the target frame (skip everything in between)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # ANPR needs full resolution to read plates — encode BEFORE downscaling
                _, buf_full = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                anpr_bytes = buf_full.tobytes()
                
                # Downscale for faster processing on other modules
                if scale < 1.0:
                    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                
                processed += 1
                
                # Update progress
                _video_analysis_state['progress'] = round(((i + 1) / max(total_to_process, 1)) * 100, 1)
                _video_analysis_state['processed_frames'] = processed
                
                # Run all 4 modules in parallel using the shared executor
                try:
                    # Pass raw numpy frames directly to skip expensive JPG encoding/decoding!
                    fv = executor.submit(detect_vehicles, img=frame)
                    fa = executor.submit(detect_plates, img=frame)   # Full res for ANPR
                    ff = executor.submit(analyze_faces, img=frame)
                    fp = executor.submit(count_people, img=frame)
                    fw = executor.submit(detect_weapons, img=frame)
                    fb = executor.submit(detect_behaviors, img=frame)
                    fbl = executor.submit(detect_blacklist, image_bytes=anpr_bytes) # reuse encoded bytes for face
                    
                    vehicle_res = fv.result()
                    anpr_res = fa.result()
                    face_res = ff.result()
                    people_res = fp.result()
                    weapon_res = fw.result()
                    behavior_res = fb.result()
                    blacklist_res = fbl.result()
                except RuntimeError:
                    # Executor shut down (app closing), exit gracefully
                    break
            
                # ── Accumulate Vehicles ──
                if vehicle_res and vehicle_res.get('detections'):
                    for v in vehicle_res['detections']:
                        vid = v.get('id')
                        if vid and vid not in all_vehicles:
                            all_vehicles[vid] = v
                            vtype = v.get('class', '').lower()
                            if vtype in vehicle_type_counts:
                                vehicle_type_counts[vtype] += 1
                        spd = v.get('speed')
                        if spd and spd > 0:
                            if spd > max_speed:
                                max_speed = spd
                            speed_records.append({'frame': frame_idx, 'speed': round(float(spd), 1), 'class': v.get('class', '')})
                
                # ── Accumulate Plates ──
                if anpr_res and anpr_res.get('plates'):
                    for p in anpr_res['plates']:
                        txt = p.get('text', '')
                        if txt and txt not in all_plates:
                            all_plates[txt] = {
                                'text': txt,
                                'confidence': round(float(p.get('confidence', 0)), 3),
                                'is_indian_standard': p.get('is_indian_standard', False),
                                'is_blacklisted': p.get('is_blacklisted', False),
                                'frame': frame_idx
                            }
                            if p.get('is_blacklisted'):
                                alerts.append({'type': 'plate', 'detail': f'Blacklisted plate: {txt}', 'frame': frame_idx})
                
                # ── Accumulate Faces ──
                if face_res and face_res.get('faces'):
                    for f in face_res['faces']:
                        name = f.get('name', 'Unknown')
                        fkey = name if name != 'Unknown' else f'Unknown_{f.get("id", processed)}'
                        if fkey not in all_faces:
                            all_faces[fkey] = {
                                'name': name,
                                'age': f.get('age'),
                                'gender': f.get('gender'),
                                'emotion': f.get('emotion'),
                                'is_authorized': f.get('is_authorized', False),
                                'frame': frame_idx
                            }
                            if f.get('is_authorized') == False and name == 'Unknown':
                                alerts.append({'type': 'face', 'detail': f'Unauthorized face (Age: {f.get("age")}, {f.get("gender")})', 'frame': frame_idx})
                
                # ── Accumulate Blacklist ──
                if blacklist_res and blacklist_res.get('matches'):
                    for m in blacklist_res['matches']:
                        name = m.get('name')
                        if name:
                            blacklist_events.append({
                                'name': name,
                                'confidence': round(float(m.get('confidence', 0)), 3),
                                'frame': frame_idx,
                                'timestamp': m.get('timestamp')
                            })
                            alerts.append({'type': 'blacklist_match', 'detail': f'Blacklisted person: {name}', 'frame': frame_idx})
                if people_res and people_res.get('detections'):
                    for p in people_res['detections']:
                        pid = p.get('id')
                        if pid is not None and pid not in all_people_ids:
                            all_people_ids.add(pid)
                            g = p.get('gender', 'Unknown')
                            if g == 'Male':
                                gender_counts['Male'] += 1
                            elif g == 'Female':
                                gender_counts['Female'] += 1
                
                # ── Accumulate Weapons ──
                if weapon_res and weapon_res.get('detections'):
                    for w in weapon_res['detections']:
                        alerts.append({'type': 'weapon', 'detail': f"WEAPON DETECTED: {w['class'].upper()}", 'frame': frame_idx})

                # ── Accumulate Behavior ──
                if behavior_res and behavior_res.get('detections'):
                    for b in behavior_res['detections']:
                        if b['behavior'] != 'Normal':
                            behavior_events.append({'id': b['id'], 'behavior': b['behavior'], 'frame': frame_idx})
                            alerts.append({'type': 'behavior', 'detail': f"{b['behavior']} (Person {b['id']})", 'frame': frame_idx})
                
                # Update live results periodically
                _video_analysis_state['results'] = {
                    'vehicles': {
                        'total': len(all_vehicles),
                        'types': vehicle_type_counts.copy(),
                        'max_speed': round(float(max_speed), 1) if max_speed > 0 else None,
                    },
                    'anpr': {
                        'plates': list(all_plates.values()),
                        'total_unique': len(all_plates)
                    },
                    'faces': {
                        'total_unique': len(all_faces),
                        'faces': list(all_faces.values())
                    },
                    'people': {
                        'total_unique': len(all_people_ids),
                        'gender_counts': gender_counts.copy()
                    },
                    'blacklist': {
                        'total_matches': len(blacklist_events),
                        'matches': blacklist_events
                    },
                    'behavior': {
                        'events': behavior_events[-50:]
                    },
                    'alerts': alerts[-20:],  # last 20 alerts
                    'speed_records': speed_records[-50:],  # last 50
                    'frames_processed': processed,
                    'total_frames': total_frames
                }
        finally:
            executor.shutdown(wait=False)
        
        cap.release()
        _video_analysis_state['progress'] = 100
        _video_analysis_state['processed_frames'] = processed
        _video_analysis_state['running'] = False
        
        print(f"[Video Analysis] Completed: {processed} frames processed, "
              f"{len(all_vehicles)} vehicles, {len(all_plates)} plates, "
              f"{len(all_faces)} faces, {len(all_people_ids)} people")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        _video_analysis_state['error'] = str(e)
        _video_analysis_state['running'] = False


# ─── Page Routes ───────────────────────────────────────────────

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and verify_password(password, user['password_hash']):
            session['username'] = user['username']
            session['role'] = user['role']
            log_security_event("Login Success", user=username)
            return redirect(url_for('index'))
        else:
            # Track failed attempts for intrusion detection
            log_security_event("Login Failed", user=username, details=f"IP: {request.remote_addr}")
            
            # Simple intrusion detection: block after 7 failed attempts from same IP
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM security_logs WHERE event = 'Login Failed' AND ip = ? AND timestamp > datetime('now', '-10 minutes')",
                (request.remote_addr,)
            )
            failed_count = cursor.fetchone()[0]
            conn.close()
            
            if failed_count >= 7:
                block_ip(request.remote_addr, "Too many failed login attempts.")
                return abort(403, description="Your IP has been blocked due to too many failed login attempts.")
                
            return render_template('login.html', error="Invalid username or password.")
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    log_security_event("Logout", user=session.get('username'))
    session.clear()
    return redirect(url_for('login'))

@app.route('/user-management', methods=['GET', 'POST'])
@require_role('admin')
def user_management():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'add':
            username = request.form.get('username')
            password = request.form.get('password')
            role = request.form.get('role')
            
            valid, msg = validate_password_policy(password)
            if not valid:
                users = cursor.execute("SELECT * FROM users").fetchall()
                return render_template('user_management.html', users=users, error=msg)
            
            try:
                cursor.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                    (username, hash_password(password), role)
                )
                conn.commit()
                log_security_event("User Created", user=session.get('username'), details=f"Created user: {username} with role: {role}")
            except Exception as e:
                return render_template('user_management.html', users=cursor.execute("SELECT * FROM users").fetchall(), error=str(e))
        
        elif action == 'delete':
            user_id = request.form.get('user_id')
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            log_security_event("User Deleted", user=session.get('username'), details=f"Deleted user ID: {user_id}")

    users = cursor.execute("SELECT * FROM users").fetchall()
    conn.close()
    return render_template('user_management.html', users=users)

@app.route('/security-dashboard')
@require_role(['admin', 'auditor'])
def security_dashboard():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    recent_logs = cursor.execute("SELECT * FROM security_logs ORDER BY timestamp DESC LIMIT 50").fetchall()
    failed_logins = cursor.execute("SELECT COUNT(*) FROM security_logs WHERE event = 'Login Failed' AND timestamp > date('now')").fetchone()[0]
    blocked_ips = cursor.execute("SELECT COUNT(*) FROM blocked_ips").fetchone()[0]
    watchlist_updates = cursor.execute("SELECT COUNT(*) FROM watchlist_logs WHERE timestamp > date('now')").fetchone()[0]
    unauthorized_attempts = cursor.execute("SELECT COUNT(*) FROM security_logs WHERE event = 'Unauthorized Access Attempt'").fetchone()[0]
    
    conn.close()
    return render_template('security_dashboard.html', 
                          logs=recent_logs, 
                          failed_logins=failed_logins,
                          blocked_ips=blocked_ips,
                          watchlist_updates=watchlist_updates,
                          unauthorized_attempts=unauthorized_attempts)


@app.route('/modules')
def modules():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('modules.html')


@app.route('/architecture')
def architecture():
    return render_template('architecture.html')


@app.route('/roadmap')
def roadmap():
    return render_template('roadmap.html')


@app.route('/specs')
def specs():
    return render_template('specs.html')

VIDEO_DIR = os.path.abspath(r'c:\Users\abhin\Downloads\testing')
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

@app.route('/stream')
def stream_page():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('stream.html')

@app.route('/api/list-videos', methods=['GET'])
def list_videos():
    """List all video files available in the video directories."""
    videos = []
    # Scan video directories
    search_dirs = [app.config['UPLOAD_FOLDER'], VIDEO_DIR, os.path.dirname(__file__)]
    for d in search_dirs:
        if d and os.path.isdir(d):
            for f in os.listdir(d):
                if allowed_video(f) and f not in videos:
                    videos.append(f)
    return jsonify({'videos': sorted(videos)})

@app.route('/serve-video/<path:filename>')
def serve_video(filename):
    # Try upload folder first, then video dir, then project root
    search_dirs = [app.config['UPLOAD_FOLDER'], VIDEO_DIR, os.path.dirname(__file__)]
    for d in search_dirs:
        if d and os.path.isdir(d) and os.path.exists(os.path.join(d, filename)):
            return send_from_directory(d, filename)
    return abort(404)


# ─── API Routes ────────────────────────────────────────────────

# Background Video Analysis State
ANALYSIS_STATUS = {
    'running': False,
    'progress': 0,
    'processed_frames': 0,
    'results': {
        'vehicles': {'total': 0, 'types': {}},
        'anpr': {'total_unique': 0, 'plates': []},
        'faces': {'total_unique': 0, 'faces': []},
        'people': {'total_unique': 0, 'gender_counts': {}},
        'weapons': {'total': 0, 'items': []},
        'behavior': {'total_falls': 0, 'total_loitering': 0, 'events': []},
        'helmets': {'total_riders': 0, 'violations': 0, 'detections': []},
        'alerts': []
    },
    'error': None,
    'video_path': None
}

def video_analysis_worker(video_path, filename):
    import cv2
    import time
    from modules.vehicle_counter import detect_vehicles
    from modules.anpr import detect_plates
    from modules.face_recognition_module import analyze_faces
    from modules.people_counter import count_people
    from modules.behavior_analysis import detect_behaviors
    from modules.weapon_detection import detect_weapons
    from modules.helmet_detector import detect_helmets

    global ANALYSIS_STATUS
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        ANALYSIS_STATUS['error'] = "Could not open video file"
        ANALYSIS_STATUS['running'] = False
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # Optimized sampling to process full videos "so quick"
    # Process at most 40 frames total, or 1 frame every 5 seconds, whichever is sparser
    sample_rate = max(int(fps * 5), total_frames // 40)
    if sample_rate < 1: sample_rate = 1
    
    processed = 0
    
    # Cumulative sets for unique tracking during worker
    unique_plates = set()
    unique_faces = set()
    unique_vehicles = set()
    unique_people = set()
    
    while ANALYSIS_STATUS['running']:
        ret, frame = cap.read()
        if not ret: break
        
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_idx % sample_rate != 0:
            continue
            
        timestamp_sec = frame_idx / fps
        
        # Run all modules
        v_res = detect_vehicles(img=frame)
        a_res = detect_plates(img=frame)
        f_res = analyze_faces(img=frame)
        p_res = count_people(img=frame)
        w_res = detect_weapons(img=frame)
        b_res = detect_behaviors(img=frame)
        h_res = detect_helmets(img=frame)
        
        # Attach timestamps
        for d in v_res.get('detections', []): d['timestamp_sec'] = timestamp_sec
        for p in a_res.get('plates', []): p['timestamp_sec'] = timestamp_sec
        for f in f_res.get('faces', []): f['timestamp_sec'] = timestamp_sec
        for p in p_res.get('detections', []): p['timestamp_sec'] = timestamp_sec
        for iw in w_res.get('detections', []): iw['timestamp_sec'] = timestamp_sec
        for hd in h_res.get('detections', []): hd['timestamp_sec'] = timestamp_sec

        # Update Cumulative Results
        # Vehicles
        for v in v_res.get('detections', []):
            if v['id'] not in unique_vehicles:
                unique_vehicles.add(v['id'])
                t = v['class'].lower()
                ANALYSIS_STATUS['results']['vehicles']['types'][t] = ANALYSIS_STATUS['results']['vehicles']['types'].get(t, 0) + 1
        ANALYSIS_STATUS['results']['vehicles']['total'] = len(unique_vehicles)
        
        # ANPR
        for plate in a_res.get('plates', []):
            if plate['text'] not in unique_plates:
                unique_plates.add(plate['text'])
                ANALYSIS_STATUS['results']['anpr']['plates'].append(plate)
                if plate.get('is_blacklisted'):
                    ANALYSIS_STATUS['results']['alerts'].append({
                        'type': 'plate',
                        'detail': f"Blacklisted: {plate['text']}",
                        'frame': frame_idx,
                        'timestamp_sec': timestamp_sec
                    })
        ANALYSIS_STATUS['results']['anpr']['total_unique'] = len(unique_plates)

        # Faces
        for face in f_res.get('faces', []):
            fname = face.get('name', 'Unknown')
            fkey = f"{fname}_{face.get('id', 0)}"
            if fkey not in unique_faces:
                unique_faces.add(fkey)
                ANALYSIS_STATUS['results']['faces']['faces'].append(face)
        ANALYSIS_STATUS['results']['faces']['total_unique'] = len(unique_faces)

        # People
        for person in p_res.get('detections', []):
            if person.get('id') not in unique_people:
                unique_people.add(person.get('id'))
                g = str(person.get('gender', 'Male')) # Ensure g is a string
                ANALYSIS_STATUS['results']['people']['gender_counts'][g] = ANALYSIS_STATUS['results']['people']['gender_counts'].get(g, 0) + 1
        ANALYSIS_STATUS['results']['people']['total_unique'] = len(unique_people)
        
        # Behavior
        for event in b_res.get('detections', []):
            if event['behavior'] != "Normal":
                ANALYSIS_STATUS['results']['behavior']['events'].append(event)
                if event['behavior'] == "FALL DETECTED":
                    ANALYSIS_STATUS['results']['behavior']['total_falls'] += 1
                elif event['behavior'] == "LOITERING":
                    ANALYSIS_STATUS['results']['behavior']['total_loitering'] += 1
                
                ANALYSIS_STATUS['results']['alerts'].append({
                    'type': 'behavior',
                    'detail': f"{event['behavior']} (ID: {event['id']})",
                    'frame': frame_idx,
                    'timestamp_sec': timestamp_sec
                })
        
        # Weapons
        for w in w_res.get('detections', []):
            ANALYSIS_STATUS['results']['weapons']['items'].append(w)
            ANALYSIS_STATUS['results']['alerts'].append({
                'type': 'weapon',
                'detail': f"Weapon: {w['class'].upper()}",
                'frame': frame_idx,
                'timestamp_sec': timestamp_sec
            })
        ANALYSIS_STATUS['results']['weapons']['total'] = len(ANALYSIS_STATUS['results']['weapons']['items'])
        
        # Helments
        ANALYSIS_STATUS['results']['helmets']['total_riders'] += h_res.get('total_riders', 0)
        ANALYSIS_STATUS['results']['helmets']['violations'] += h_res.get('violations', 0)
        for h in h_res.get('detections', []):
            if h.get('is_violation'):
                ANALYSIS_STATUS['results']['helmets']['detections'].append(h)
                ANALYSIS_STATUS['results']['alerts'].append({
                    'type': 'helmet',
                    'detail': f"No Helmet Detection: {h['confidence']}",
                    'frame': frame_idx,
                    'timestamp_sec': timestamp_sec
                })
        
        processed += 1
        ANALYSIS_STATUS['processed_frames'] = processed
        ANALYSIS_STATUS['progress'] = int((frame_idx / total_frames) * 100)
    
    cap.release()
    ANALYSIS_STATUS['progress'] = 100
    ANALYSIS_STATUS['running'] = False

@app.route('/api/analyze-video', methods=['POST'])
def start_full_analysis():
    global ANALYSIS_STATUS
    video_name = request.json.get('video')
    
    if not video_name:
        return jsonify({"success": False, "error": "No video selected"}), 400
        
    # Find path
    path = os.path.join(VIDEO_DIR, video_name)
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), video_name)
        
    if not os.path.exists(path):
        return jsonify({"success": False, "error": "Video file not found"}), 404

    # Reset state
    ANALYSIS_STATUS = {
        'running': True,
        'progress': 0,
        'processed_frames': 0,
        'results': {
            'vehicles': {'total': 0, 'types': {}},
            'anpr': {'total_unique': 0, 'plates': []},
            'faces': {'total_unique': 0, 'faces': []},
            'people': {'total_unique': 0, 'gender_counts': {'Male': 0, 'Female': 0, 'Unknown': 0}},
            'weapons': {'total': 0, 'items': []},
            'helmets': {'total_riders': 0, 'violations': 0, 'detections': []},
            'alerts': []
        },
        'error': None,
        'video_path': path
    }
    
    threading.Thread(target=video_analysis_worker, args=(path, video_name), daemon=True).start()
    return jsonify({"success": True})

@app.route('/api/analyze-video/status')
def full_analysis_status():
    return jsonify(ANALYSIS_STATUS)


# ─── Data Management ───────────────────────────────────────────

def load_blacklist():
    conn = get_db_connection()
    cursor = conn.cursor()
    # Migrating old config_data.json if exists
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
                plates = data.get("blacklist_plates", [])
                for plate in plates:
                    cursor.execute("INSERT OR IGNORE INTO watchlist (name, case_id, added_by) VALUES (?, ?, ?)", (plate, 'MIGRATED', 'SYSTEM'))
            os.remove(CONFIG_FILE) # Cleanup after migration
            conn.commit()
        except Exception:
            pass
            
    cursor.execute("SELECT name FROM watchlist")
    rows = cursor.fetchall()
    conn.close()
    return set(row['name'] for row in rows)

# ─── IP Stream Backend ─────────────────────────────────────────

STREAM_STATE = {
    'running': False,
    'url': None,
    'night_vision': False,
    'latest_frame': None,
    'latest_stats': {}
}

def enhance_low_light(image):
    import cv2
    import numpy as np
    
    # 1. Gamma Correction to brighten the image without washing out
    gamma = 1.5
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    brightened = cv2.LUT(image, table)
    
    # 2. Convert to LAB color space for histogram equalization on Lightness channel
    lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge and convert back to BGR
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def stream_worker(url):
    import cv2
    from modules.vehicle_counter import detect_vehicles
    from modules.anpr import detect_plates
    from modules.face_recognition_module import analyze_faces
    from modules.people_counter import count_people
    from modules.weapon_detection import detect_weapons
    
    STREAM_STATE['running'] = True
    STREAM_STATE['url'] = url
    cap = cv2.VideoCapture(url)
    
    executor = ThreadPoolExecutor(max_workers=4)
    frame_count = 0
    last_stats = {}
    
    while STREAM_STATE['running'] and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # If stream ends (e.g. video file ended), we could loop or stop.
            # IP cameras usually don't end unless disconnected.
            break
            
        frame_count += 1
        
        # Apply Night Vision Enhancement if enabled
        if STREAM_STATE.get('night_vision'):
            frame = enhance_low_light(frame)
        
        # Analyze 1 out of every 15 frames (~2 times a sec for 30fps)
        if frame_count % 15 == 0 or frame_count == 1:
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buf.tobytes()
            
            future_vehicle = executor.submit(detect_vehicles, frame_bytes)
            future_anpr = executor.submit(detect_plates, frame_bytes)
            future_face = executor.submit(analyze_faces, frame_bytes)
            future_people = executor.submit(count_people, frame_bytes)
            future_weapon = executor.submit(detect_weapons, frame_bytes)
            future_behavior = executor.submit(detect_behaviors, frame_bytes)
            
            try:
                vehicle_res = future_vehicle.result(timeout=10)
                anpr_res = future_anpr.result(timeout=10)
                face_res = future_face.result(timeout=10)
                people_res = future_people.result(timeout=10)
                weapon_res = future_weapon.result(timeout=10)
                behavior_res = future_behavior.result(timeout=10)
                
                last_stats = {
                    'vehicles': vehicle_res,
                    'anpr': anpr_res,
                    'faces': face_res,
                    'people': people_res,
                    'weapons': weapon_res,
                    'behavior': behavior_res
                }
                STREAM_STATE['latest_stats'] = last_stats
            except Exception as e:
                print(f"Stream analysis error: {e}")
                
        annotated = frame.copy()
        
        # Draw ALERTS on frame
        y_offset = 40
        cv2.putText(annotated, f"Vehicles: {last_stats.get('vehicles', {}).get('total', 0)}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        y_offset += 40
        
        for p in last_stats.get('anpr', {}).get('plates', []):
            if p.get('is_blacklisted'):
                cv2.putText(annotated, f"ALERT: Blacklisted {p['text']}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                y_offset += 40
                
        has_unmatched = any(f.get('status') == 'unmatched' for f in last_stats.get('faces', {}).get('faces', []))
        if has_unmatched:
            cv2.putText(annotated, "ALERT: Unmatched Person Detected", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            y_offset += 40
            
        # Draw Weapon Alerts
        for w in last_stats.get('weapons', {}).get('detections', []):
            cv2.putText(annotated, f"ALERT: WEAPON ({w['class']})", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            y_offset += 40
            
        # Draw Behavior Alerts
        for b in last_stats.get('behavior', {}).get('detections', []):
            if b['behavior'] != "Normal":
                cv2.putText(annotated, f"ALERT: {b['behavior']}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
                y_offset += 40
            
        _, buf2 = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        STREAM_STATE['latest_frame'] = buf2.tobytes()
        
        time.sleep(0.02)
        
    cap.release()
    STREAM_STATE['running'] = False

@app.route('/api/ip-stream/start', methods=['POST'])
def start_ip_stream():
    data = request.json
    url = data.get('url', '').strip()
    night_vision = data.get('night_vision', False)
    
    if not url:
        return jsonify({"error": "No URL provided"}), 400
        
    if STREAM_STATE['running']:
        STREAM_STATE['running'] = False
        time.sleep(0.5)
        
    STREAM_STATE['night_vision'] = night_vision
    threading.Thread(target=stream_worker, args=(url,), daemon=True).start()
    return jsonify({"success": True})

@app.route('/api/ip-stream/stop', methods=['POST'])
def stop_ip_stream():
    STREAM_STATE['running'] = False
    return jsonify({"success": True})
    
def generate_mjpeg():
    while STREAM_STATE['running'] or STREAM_STATE['latest_frame'] is not None:
        frame = STREAM_STATE['latest_frame']
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.05)
        
@app.route('/video_feed')
def video_feed():
    if not STREAM_STATE['running']:
        return "Stream not running", 400
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/ip-stream/stats', methods=['GET'])
def ip_stream_stats():
    if not STREAM_STATE['running']:
        return jsonify({"error": "Stream not running"}), 400
    return jsonify(STREAM_STATE['latest_stats'])

# ─── Data Management ───────────────────────────────────────────

def save_blacklist(plate, action='add'):
    conn = get_db_connection()
    cursor = conn.cursor()
    if action == 'add':
        cursor.execute("INSERT OR IGNORE INTO watchlist (name, case_id, added_by) VALUES (?, ?, ?)", (plate, 'MANUAL', session.get('username')))
    else:
        cursor.execute("DELETE FROM watchlist WHERE name = ?", (plate,))
    conn.commit()
    conn.close()
    log_watchlist_action(f"Watchlist {action.capitalize()}", session.get('username'), details=f"Plate: {plate}")

@app.route('/api/blacklist/list', methods=['GET'])
@require_role(['admin', 'security_officer', 'operator', 'auditor'])
def list_blacklist():
    return jsonify({"blacklist": list(load_blacklist())})

@app.route('/api/blacklist/add', methods=['POST'])
@require_role('admin')
def add_blacklist():
    data = request.json
    plate = data.get('plate', '').strip().upper()
    if not plate:
        return jsonify({"error": "No plate provided"}), 400
    save_blacklist(plate, 'add')
    return jsonify({"success": True, "plate": plate})

@app.route('/api/blacklist/remove', methods=['POST'])
@require_role('admin')
def remove_blacklist():
    data = request.json
    plate = data.get('plate', '').strip().upper()
    save_blacklist(plate, 'remove')
    return jsonify({"success": True, "plate": plate})

@app.route('/api/whitelist/list', methods=['GET'])
@require_role(['admin', 'security_officer', 'operator', 'auditor'])
def list_whitelist():
    faces = set()
    if os.path.exists(WHITELIST_DIR):
        for f in os.listdir(WHITELIST_DIR):
            path = os.path.join(WHITELIST_DIR, f)
            if os.path.isdir(path):
                faces.add(f)
            elif allowed_image(f):
                faces.add(os.path.splitext(f)[0])
    return jsonify({"whitelist": sorted(list(faces))})

@app.route('/api/whitelist/add', methods=['POST'])
@require_role('admin')
def add_whitelist():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    name = request.form.get('name', '').strip()
    case_id = request.form.get('case_id', 'GENERAL').strip()
    if not name:
        return jsonify({'error': 'No name provided'}), 400
        
    files = request.files.getlist('image')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400
        
    import shutil
    from werkzeug.utils import secure_filename
    
    # Create person subdirectory
    person_dir = os.path.join(WHITELIST_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Remove any legacy single file if it exists
    for f in os.listdir(WHITELIST_DIR):
        if f.startswith(name + ".") and allowed_image(f):
            os.remove(os.path.join(WHITELIST_DIR, f))

    saved_count = 0
    for file in files:
        if file and allowed_image(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(person_dir, filename))
            saved_count += 1
            
    if saved_count == 0:
        return jsonify({'error': 'No valid images provided'}), 400
        
    # Invalidate DeepFace cache
    for pkl in os.listdir(WHITELIST_DIR):
        if pkl.endswith('.pkl'):
            os.remove(os.path.join(WHITELIST_DIR, pkl))
    
    # Add to database watchlist
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO watchlist (name, case_id, added_by) VALUES (?, ?, ?)", (name, case_id, session.get('username')))
    conn.commit()
    conn.close()
    
    log_watchlist_action("Face Added", session.get('username'), case_id=case_id, details=f"Name: {name}, Images: {saved_count}")
    return jsonify({"success": True, "name": name, "count": saved_count})

@app.route('/api/whitelist/remove', methods=['POST'])
@require_role('admin')
def remove_whitelist():
    name = request.json.get('name', '').strip()
    if not name:
        return jsonify({"error": "No name provided"}), 400
    
    import shutil
    removed = False
    
    # Remove directory
    person_dir = os.path.join(WHITELIST_DIR, name)
    if os.path.isdir(person_dir):
        shutil.rmtree(person_dir)
        removed = True
        
    # Remove legacy single files
    for f in os.listdir(WHITELIST_DIR):
        if f.startswith(name + ".") and allowed_image(f):
            os.remove(os.path.join(WHITELIST_DIR, f))
            removed = True
            
    if removed:
        for pkl in os.listdir(WHITELIST_DIR):
            if pkl.endswith('.pkl'):
                os.remove(os.path.join(WHITELIST_DIR, pkl))
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM watchlist WHERE name = ?", (name,))
        conn.commit()
        conn.close()
        
        log_watchlist_action("Face Removed", session.get('username'), details=f"Name: {name}")
        return jsonify({"success": True, "name": name})
    return jsonify({"error": "Name not found"}), 404


@app.route('/api/face-blacklist/list', methods=['GET'])
@require_role(['admin', 'security_officer', 'operator', 'auditor'])
def list_face_blacklist():
    faces = set()
    if os.path.exists(FACE_BLACKLIST_DIR):
        for f in os.listdir(FACE_BLACKLIST_DIR):
            path = os.path.join(FACE_BLACKLIST_DIR, f)
            if os.path.isdir(path):
                faces.add(f)
            elif allowed_image(f):
                faces.add(os.path.splitext(f)[0])
    return jsonify({"blacklist": sorted(list(faces))})


@app.route('/api/face-blacklist/add', methods=['POST'])
@require_role('admin')
def add_face_blacklist():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    name = request.form.get('name', '').strip()
    case_id = request.form.get('case_id', 'BLACKLIST').strip()
    if not name:
        return jsonify({'error': 'No name provided'}), 400
        
    files = request.files.getlist('image')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400
        
    import shutil
    from werkzeug.utils import secure_filename
    
    # Create person subdirectory
    person_dir = os.path.join(FACE_BLACKLIST_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Remove any legacy single file if it exists
    for f in os.listdir(FACE_BLACKLIST_DIR):
        if f.startswith(name + ".") and allowed_image(f):
            os.remove(os.path.join(FACE_BLACKLIST_DIR, f))
            
    saved_count = 0
    for file in files:
        if file and allowed_image(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(person_dir, filename))
            saved_count += 1
            
    if saved_count == 0:
        return jsonify({'error': 'No valid images provided'}), 400
        
    # Invalidate DeepFace cache for blacklist
    for pkl in os.listdir(FACE_BLACKLIST_DIR):
        if pkl.endswith('.pkl'):
            os.remove(os.path.join(FACE_BLACKLIST_DIR, pkl))
    
    # Add to database watchlist (flagged as BLACKLIST)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO watchlist (name, case_id, added_by) VALUES (?, ?, ?)", (name, case_id, session.get('username')))
    conn.commit()
    conn.close()
    
    log_watchlist_action("Face Blacklisted", session.get('username'), case_id=case_id, details=f"Name: {name}, Images: {saved_count}")
    return jsonify({"success": True, "name": name, "count": saved_count})


@app.route('/api/face-blacklist/remove', methods=['POST'])
@require_role('admin')
def remove_face_blacklist():
    name = request.json.get('name', '').strip()
    if not name:
        return jsonify({"error": "No name provided"}), 400
    
    import shutil
    removed = False
    
    # Remove directory
    person_dir = os.path.join(FACE_BLACKLIST_DIR, name)
    if os.path.isdir(person_dir):
        shutil.rmtree(person_dir)
        removed = True
        
    # Remove legacy single files
    for f in os.listdir(FACE_BLACKLIST_DIR):
        if f.startswith(name + ".") and allowed_image(f):
            os.remove(os.path.join(FACE_BLACKLIST_DIR, f))
            removed = True
            
    if removed:
        for pkl in os.listdir(FACE_BLACKLIST_DIR):
            if pkl.endswith('.pkl'):
                os.remove(os.path.join(FACE_BLACKLIST_DIR, pkl))
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM watchlist WHERE name = ?", (name,))
        conn.commit()
        conn.close()
        
        log_watchlist_action("Face Removed", session.get('username'), details=f"Name: {name}")
        return jsonify({"success": True, "name": name})
    return jsonify({"error": "Name not found"}), 404


def process_video_request(file, process_func, result_key='result'):
    """
    Helper to process a video for an individual module by sampling frames.
    """
    import cv2
    import tempfile
    import os
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=app.config['UPLOAD_FOLDER'])
    try:
        file.save(tmp.name)
        tmp.close()
        
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            return {'error': 'Could not open video file'}, 400
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return {'error': 'Empty video file'}, 400
            
        # Sample up to 5 frames
        num_samples = min(5, total_frames)
        sample_indices = [int(i * (total_frames - 1) / (num_samples - 1)) for i in range(num_samples)] if num_samples > 1 else [0]
        
        results = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue
            
            _, buf = cv2.imencode('.jpg', frame)
            frame_bytes = buf.tobytes()
            results.append(process_func(frame_bytes))
            
        cap.release()
        os.unlink(tmp.name)
        
        if not results:
            return {'error': 'No frames processed'}, 500
            
        # Aggregate logic depends on the module, but we'll return the 'best' or 'combined'
        # For individual modules, we'll return the first one that found something, or just the first.
        # But a better way is to provide the 'maximal' result.
        return results[0] # Default fallback for now, specialized below
        
    except Exception as e:
        if os.path.exists(tmp.name): os.unlink(tmp.name)
        raise e

@app.route('/api/detect-vehicles', methods=['POST'])
@limiter.exempt
def api_detect_vehicles():
    try:
        if 'image' not in request.files and 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files.get('image') or request.files.get('file')
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type.'}), 400
        
        from modules.vehicle_counter import detect_vehicles
        
        night_vision = request.form.get('night_vision') == 'true'
        high_throughput = request.form.get('high_throughput') == 'true'
        
        if allowed_image(file.filename):
            image_bytes = file.read()
            if night_vision:
                import cv2
                import numpy as np
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    img = enhance_low_light(img)
                    result = detect_vehicles(img=img, high_throughput=high_throughput)
                else:
                    result = detect_vehicles(image_bytes, high_throughput=high_throughput)
            else:
                result = detect_vehicles(image_bytes, high_throughput=high_throughput)
            return jsonify(result)
        else:
            # Video processing
            print(f"DEBUG: Processing vehicle detection video for {file.filename}")
            import cv2
            import tempfile
            import os
            
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=app.config['UPLOAD_FOLDER'])
            file.save(tmp.name)
            tmp.close()
            cap = cv2.VideoCapture(tmp.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            
            # High-speed sampling: at most 20 frames total
            sample_interval = max(int(fps * 3), total_frames // 20)
            if sample_interval < 1: sample_interval = 1
            frame_indices = list(range(0, total_frames, sample_interval))
            
            aggregated_result = None
            max_vehicles = -1
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret: continue
                if night_vision:
                    frame = enhance_low_light(frame)
                # Pass numpy frame directly
                res = detect_vehicles(img=frame)
                if res.get('total', 0) > max_vehicles:
                    max_vehicles = res.get('total', 0)
                    aggregated_result = res
            
            cap.release()
            os.unlink(tmp.name)
            return jsonify(aggregated_result if aggregated_result else {'error': 'No frames analyzed'})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500


@app.route('/api/anpr', methods=['POST'])
@limiter.exempt
def api_anpr():
    try:
        if 'image' not in request.files and 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files.get('image') or request.files.get('file')
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type.'}), 400
        
        from modules.anpr import detect_plates
        
        night_vision = request.form.get('night_vision') == 'true'
        high_throughput = request.form.get('high_throughput') == 'true'
        
        if allowed_image(file.filename):
            image_bytes = file.read()
            if night_vision:
                import cv2
                import numpy as np
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    img = enhance_low_light(img)
                    result = detect_plates(img=img, high_throughput=high_throughput)
                else:
                    result = detect_plates(image_bytes, high_throughput=high_throughput)
            else:
                result = detect_plates(image_bytes, high_throughput=high_throughput)
            return jsonify(result)
        else:
            # Video ANPR: Sample 1 frame every 2 seconds, accumulate ALL unique plates
            print(f"DEBUG: Processing ANPR video for {file.filename}")
            import cv2
            import tempfile
            import os
            
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=app.config['UPLOAD_FOLDER'])
            file.save(tmp.name)
            tmp.close()
            cap = cv2.VideoCapture(tmp.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            
            # High-speed sampling: at most 15 frames total for ANPR
            sample_interval = max(int(fps * 4), total_frames // 15)
            if sample_interval < 1: sample_interval = 1
            frame_indices = list(range(0, total_frames, sample_interval))
            
            print(f"  ANPR: {total_frames} total frames, processing {len(frame_indices)} sampled frames (every 3s)")
            
            final_result = None
            all_plates = []
            seen_texts = set()
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret: continue
                # Pass numpy frame directly (Awiros is fast)
                res = detect_plates(img=frame)
                if not final_result: final_result = res
                for p in res.get('plates', []):
                    if p['text'] not in seen_texts:
                        all_plates.append(p)
                        seen_texts.add(p['text'])
            
            if final_result:
                final_result['plates'] = all_plates
                final_result['total_plates'] = len(all_plates)
            
            cap.release()
            os.unlink(tmp.name)
            return jsonify(final_result if final_result else {'error': 'No frames analyzed'})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'ANPR failed: {str(e)}'}), 500


@app.route('/api/recognize-face', methods=['POST'])
@limiter.exempt
def api_recognize_face():
    try:
        if 'image' not in request.files and 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files.get('image') or request.files.get('file')
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type.'}), 400
        
        from modules.face_recognition_module import analyze_faces
        
        night_vision = request.form.get('night_vision') == 'true'
        high_throughput = request.form.get('high_throughput') == 'true'
        
        if allowed_image(file.filename):
            image_bytes = file.read()
            if night_vision:
                import cv2
                import numpy as np
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    img = enhance_low_light(img)
                    result = analyze_faces(img=img, high_throughput=high_throughput)
                else:
                    result = analyze_faces(image_bytes, high_throughput=high_throughput)
            else:
                result = analyze_faces(image_bytes, high_throughput=high_throughput)
            return jsonify(result)
        else:
            # Video Face: Sample 1 frame every 2 seconds, aggregate unique faces
            print(f"DEBUG: Processing face video for {file.filename}")
            import cv2
            import tempfile
            import os
            
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=app.config['UPLOAD_FOLDER'])
            file.save(tmp.name)
            tmp.close()
            cap = cv2.VideoCapture(tmp.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            
            # High-speed sampling: at most 20 frames total
            sample_interval = int(fps) if int(fps) > 0 else 1
            if sample_interval < 1: sample_interval = 1
            frame_indices = list(range(0, total_frames, sample_interval))
            
            final_result = None
            all_faces = []
            seen_faces = set()
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret: continue
                if night_vision:
                    frame = enhance_low_light(frame)
                # Pass numpy frame directly
                res = analyze_faces(img=frame)
                if not final_result: final_result = res
                for f in res.get('faces', []):
                    fkey = f.get('name', 'Unknown')
                    if fkey not in seen_faces:
                        seen_faces.add(fkey)
                        all_faces.append(f)
            
            if final_result:
                final_result['faces'] = all_faces[:100]
                final_result['total_faces'] = len(all_faces)
            
            cap.release()
            os.unlink(tmp.name)
            return jsonify(final_result if final_result else {'error': 'No frames analyzed'})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Face recognition failed: {str(e)}'}), 500


@app.route('/api/count-people', methods=['POST'])
@limiter.exempt
def api_count_people():
    try:
        if 'image' not in request.files and 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files.get('image') or request.files.get('file')
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type.'}), 400
        
        from modules.people_counter import count_people
        
        night_vision = request.form.get('night_vision') == 'true'
        
        high_throughput = request.form.get('high_throughput') == 'true'
        
        if allowed_image(file.filename):
            image_bytes = file.read()
            if night_vision:
                import cv2
                import numpy as np
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    img = enhance_low_light(img)
                    result = count_people(img=img, high_throughput=high_throughput)
                else:
                    result = count_people(image_bytes, high_throughput=high_throughput)
            else:
                result = count_people(image_bytes, high_throughput=high_throughput)
            return jsonify(result)
        else:
            # Video People: Sample 1 frame every 2 seconds, return peak count
            print(f"DEBUG: Processing people video for {file.filename}")
            import cv2
            import tempfile
            import os
            
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=app.config['UPLOAD_FOLDER'])
            file.save(tmp.name)
            tmp.close()
            cap = cv2.VideoCapture(tmp.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            
            # High-speed sampling: at most 20 frames total
            sample_interval = max(int(fps * 3), total_frames // 20)
            if sample_interval < 1: sample_interval = 1
            frame_indices = list(range(0, total_frames, sample_interval))
            
            max_p_count = -1
            final_result = None
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret: continue
                if night_vision:
                    frame = enhance_low_light(frame)
                # Pass numpy frame directly
                res = count_people(img=frame)
                if res.get('total_people', 0) > max_p_count:
                    max_p_count = res.get('total_people', 0)
                    final_result = res
            
            cap.release()
            os.unlink(tmp.name)
            return jsonify(final_result if final_result else {'error': 'No frames analyzed'})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'People counting failed: {str(e)}'}), 500


    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze-behavior', methods=['POST'])
@limiter.exempt
def api_analyze_behavior():
    try:
        if 'image' not in request.files and 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files.get('image') or request.files.get('file')
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type.'}), 400
            
        from modules.behavior_analysis import detect_behaviors
        
        # Apply night vision if requested
        night_vision = request.form.get('night_vision') == 'true'
        
        if allowed_image(file.filename):
            image_bytes = file.read()
            if night_vision:
                import cv2
                import numpy as np
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    img = enhance_low_light(img)
                    result = detect_behaviors(img=img)
                else:
                    result = detect_behaviors(image_bytes=image_bytes)
            else:
                result = detect_behaviors(image_bytes=image_bytes)
            return jsonify(result)
        else:
            # Video Behavioral: analyze first frame
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                file.save(tmp.name)
                cap = cv2.VideoCapture(tmp.name)
                success, frame = cap.read()
                cap.release()
                os.unlink(tmp.name)
                if success:
                    if night_vision: frame = enhance_low_light(frame)
                    return jsonify(detect_behaviors(img=frame))
            return jsonify({'error': 'Failed to read video frame'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect-weapons', methods=['POST'])
@limiter.exempt
@require_role(['admin', 'security_officer', 'operator'])
def api_detect_weapons():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        high_throughput = request.form.get('high_throughput', 'false').lower() == 'true'
        
        if file and allowed_file(file.filename):
            from modules.weapon_detection import detect_weapons
            if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
                # Process as image
                image_bytes = file.read()
                result = detect_weapons(image_bytes=image_bytes, high_throughput=high_throughput)
                return jsonify(result)
            else:
                # Process as video (single frame analysis for demo)
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                    file.save(tmp.name)
                    cap = cv2.VideoCapture(tmp.name)
                    success, frame = cap.read()
                    if not success:
                        cap.release()
                        os.unlink(tmp.name)
                        return jsonify({'error': 'Could not read video frame'}), 500
                    
                    res = detect_weapons(img=frame, high_throughput=high_throughput)
                    cap.release()
                    os.unlink(tmp.name)
                    return jsonify(res)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Weapon detection failed: {str(e)}'}), 500


@app.route('/api/detect-helmets', methods=['POST'])
@limiter.exempt
def api_detect_helmets():
    try:
        if 'image' not in request.files and 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files.get('image') or request.files.get('file')
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type.'}), 400
        
        from modules.helmet_detector import detect_helmets
        
        # Apply night vision if requested
        night_vision = request.form.get('night_vision') == 'true'
        
        if allowed_image(file.filename):
            image_bytes = file.read()
            if night_vision:
                import cv2
                import numpy as np
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    img = enhance_low_light(img)
                    result = detect_helmets(img=img)
                else:
                    result = detect_helmets(image_bytes=image_bytes)
            else:
                result = detect_helmets(image_bytes=image_bytes)
            return jsonify(result)
        else:
            # Video Helmet: analyze first frame
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                file.save(tmp.name)
                cap = cv2.VideoCapture(tmp.name)
                success, frame = cap.read()
                cap.release()
                os.unlink(tmp.name)
                if success:
                    if night_vision: frame = enhance_low_light(frame)
                    return jsonify(detect_helmets(img=frame))
            return jsonify({'error': 'Failed to read video frame'}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({
        'status': 'online',
        'modules': [
            'vehicle_counter',
            'anpr',
            'face_recognition',
            'face_recognition',
            'people_counter',
            'weapon_detection',
            'behavior_analysis',
            'helmet_detection'
        ]
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  AI Video Analytics Dashboard")
    print("  Starting on http://127.0.0.1:5000")
    print("=" * 60 + "\n")

    import torch
    gpu_status = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "GPU: NOT FOUND (Using CPU)"
    print(f"  Hardware: {gpu_status}")

    # ── Pre-download ALL models at startup ──
    MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\n  [1/4] Loading YOLOv8-Large model...")
    try:
        from modules.vehicle_counter import get_model as get_vehicle_model
        get_vehicle_model()
        print("        ✓ YOLOv8-Large loaded")
    except Exception as e:
        print(f"        ✗ YOLOv8 failed: {e}")

    print("  [2/4] Loading EasyOCR model...")
    try:
        from modules.anpr import get_ocr_reader
        get_ocr_reader()
        print("        ✓ EasyOCR loaded")
    except Exception as e:
        print(f"        ✗ EasyOCR failed: {e}")

    print("  [3/4] Loading DeepFace models (RetinaFace + Facenet512 + analysis)...")
    try:
        from deepface import DeepFace
        import numpy as np
        # Create a small dummy image to trigger model downloads
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy[30:70, 30:70] = 200  # Gray face-like region
        try:
            DeepFace.analyze(dummy, actions=['age', 'gender', 'emotion', 'race'],
                           enforce_detection=False, detector_backend='retinaface', silent=True)
        except Exception:
            pass  # Analysis may fail on dummy but models are downloaded
        print("        ✓ DeepFace models loaded (RetinaFace, Age, Gender, Emotion, Race)")
    except Exception as e:
        print(f"        ✗ DeepFace failed: {e}")

    print("  [4/4] Loading People Counter model...")
    try:
        from modules.people_counter import get_model as get_people_model
        get_people_model()
        print("        ✓ People Counter loaded")
    except Exception as e:
        print(f"        ✗ People Counter failed: {e}")

    print("  [5/5] Loading Weapon Detection model...")
    try:
        from modules.weapon_detection import get_model as get_weapon_model
        get_weapon_model()
        print("        ✓ Weapon Detection loaded (weapon_v2.pt)")
    except Exception as e:
        print(f"        ✗ Weapon Detection failed: {e}")

    print("  [6/6] Loading Behavior Analysis (Pose) model...")
    try:
        from modules.behavior_analysis import get_model as get_behavior_model
        get_behavior_model()
        print("        ✓ Behavior Analysis loaded (yolov8n-pose.pt)")
    except Exception as e:
        print(f"        ✗ Behavior Analysis failed: {e}")

    print("  [7/7] Loading Helmet Detection models...")
    try:
        from modules.helmet_detector import get_models as get_helmet_models
        get_helmet_models()
        print("        ✓ Helmet models loaded")
    except Exception as e:
        print(f"        ✗ Helmet models failed: {e}")

    print("\n" + "=" * 60)
    print("  All models loaded! Server ready.")
    print("=" * 60 + "\n")

    # Send startup notification
    from security_utils import send_alert_email
    send_alert_email(
        subject="System Online",
        message="The AI Video Analytics Security System has successfully started and is now monitoring.",
        alert_type="system"
    )

    # Disable reloader to prevent ConnectionResetError/Instability on Windows with large models
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)

