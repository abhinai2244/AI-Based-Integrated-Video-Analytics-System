"""
AI Video Analytics Dashboard - Flask Application
Serves the web dashboard and provides API endpoints for AI inference.
"""

import os
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def allowed_video(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


@app.route('/api/analyze-frame', methods=['POST'])
def api_analyze_frame():
    """Lightweight endpoint for live frame analysis from browser video stream."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        image_bytes = file.read()
        
        from modules.vehicle_counter import detect_vehicles
        from modules.anpr import detect_plates
        from modules.face_recognition_module import analyze_faces
        from modules.people_counter import count_people
        
        from concurrent.futures import ThreadPoolExecutor
        
        # Run AI modules in parallel to leverage multi-core CPU/GPU
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_vehicle = executor.submit(detect_vehicles, image_bytes)
            future_anpr = executor.submit(detect_plates, image_bytes)
            future_face = executor.submit(analyze_faces, image_bytes)
            future_people = executor.submit(count_people, image_bytes)
            
            vehicle_res = future_vehicle.result()
            anpr_res = future_anpr.result()
            face_res = future_face.result()
            people_res = future_people.result()
        
        return jsonify({
            'vehicles': vehicle_res,
            'anpr': anpr_res,
            'faces': face_res,
            'people': people_res
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─── Page Routes ───────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/modules')
def modules():
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

@app.route('/stream')
def stream_page(): return render_template('stream.html')

@app.route('/serve-video/<path:filename>')
def serve_video(filename):
    return send_from_directory(os.path.abspath(r'c:\Users\abhin\Downloads\testing'), filename)


# ─── API Routes ────────────────────────────────────────────────

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
def api_detect_vehicles():
    try:
        if 'image' not in request.files and 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files.get('image') or request.files.get('file')
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type.'}), 400
        
        from modules.vehicle_counter import detect_vehicles
        
        if allowed_image(file.filename):
            image_bytes = file.read()
            print(f"DEBUG: Processing vehicle detection for {file.filename} ({len(image_bytes)} bytes)")
            result = detect_vehicles(image_bytes)
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
            num_samples = min(5, total_frames)
            sample_indices = [int(i * (total_frames - 1) / (num_samples - 1)) for i in range(num_samples)] if num_samples > 1 else [0]
            
            aggregated_result = None
            max_vehicles = -1
            
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret: continue
                _, buf = cv2.imencode('.jpg', frame)
                res = detect_vehicles(buf.tobytes())
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
def api_anpr():
    try:
        if 'image' not in request.files and 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files.get('image') or request.files.get('file')
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type.'}), 400
        
        from modules.anpr import detect_plates
        
        if allowed_image(file.filename):
            image_bytes = file.read()
            print(f"DEBUG: Processing ANPR for {file.filename} ({len(image_bytes)} bytes)")
            result = detect_plates(image_bytes)
            return jsonify(result)
        else:
            # Video ANPR: Take all unique plates found
            print(f"DEBUG: Processing ANPR video for {file.filename}")
            import cv2
            import tempfile
            import os
            
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=app.config['UPLOAD_FOLDER'])
            file.save(tmp.name)
            tmp.close()
            cap = cv2.VideoCapture(tmp.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            num_samples = min(8, total_frames) # ANPR needs more samples
            sample_indices = [int(i * (total_frames - 1) / (num_samples - 1)) for i in range(num_samples)] if num_samples > 1 else [0]
            
            final_result = None
            all_plates = []
            seen_texts = set()
            
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret: continue
                _, buf = cv2.imencode('.jpg', frame)
                res = detect_plates(buf.tobytes())
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
def api_recognize_face():
    try:
        if 'image' not in request.files and 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files.get('image') or request.files.get('file')
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type.'}), 400
        
        from modules.face_recognition_module import analyze_faces
        
        if allowed_image(file.filename):
            image_bytes = file.read()
            print(f"DEBUG: Processing face recognition for {file.filename} ({len(image_bytes)} bytes)")
            result = analyze_faces(image_bytes)
            return jsonify(result)
        else:
            # Video Face: Aggregate unique faces (simulated by ID)
            print(f"DEBUG: Processing face video for {file.filename}")
            import cv2
            import tempfile
            import os
            
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=app.config['UPLOAD_FOLDER'])
            file.save(tmp.name)
            tmp.close()
            cap = cv2.VideoCapture(tmp.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            num_samples = min(5, total_frames)
            sample_indices = [int(i * (total_frames - 1) / (num_samples - 1)) for i in range(num_samples)] if num_samples > 1 else [0]
            
            final_result = None
            all_faces = []
            
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret: continue
                _, buf = cv2.imencode('.jpg', frame)
                res = analyze_faces(buf.tobytes())
                if not final_result: final_result = res
                all_faces.extend(res.get('faces', []))
            
            if final_result:
                final_result['faces'] = all_faces[:10] # Top 10 faces across frames
                final_result['total_faces'] = len(all_faces)
            
            cap.release()
            os.unlink(tmp.name)
            return jsonify(final_result if final_result else {'error': 'No frames analyzed'})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Face recognition failed: {str(e)}'}), 500


@app.route('/api/count-people', methods=['POST'])
def api_count_people():
    try:
        if 'image' not in request.files and 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files.get('image') or request.files.get('file')
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type.'}), 400
        
        from modules.people_counter import count_people
        
        if allowed_image(file.filename):
            image_bytes = file.read()
            print(f"DEBUG: Processing people counting for {file.filename} ({len(image_bytes)} bytes)")
            result = count_people(image_bytes)
            return jsonify(result)
        else:
            # Video People: Return peak count and highest density
            print(f"DEBUG: Processing people video for {file.filename}")
            import cv2
            import tempfile
            import os
            
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=app.config['UPLOAD_FOLDER'])
            file.save(tmp.name)
            tmp.close()
            cap = cv2.VideoCapture(tmp.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            num_samples = min(5, total_frames)
            sample_indices = [int(i * (total_frames - 1) / (num_samples - 1)) for i in range(num_samples)] if num_samples > 1 else [0]
            
            max_p_count = -1
            final_result = None
            
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret: continue
                _, buf = cv2.imencode('.jpg', frame)
                res = count_people(buf.tobytes())
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


# ─── Video Analysis API ───────────────────────────────────────

@app.route('/api/analyze-video', methods=['POST'])
def api_analyze_video():
    """
    Accepts a video upload, extracts key frames, and runs all 4 AI modules.
    Returns aggregated results from vehicle detection, ANPR, face recognition,
    and people counting across sampled frames.
    """
    import cv2
    import numpy as np
    import base64
    import tempfile
    import traceback

    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'}), 400

    file = request.files['video']
    if not file or not allowed_video(file.filename):
        return jsonify({'error': 'Invalid file type. Upload a video (MP4, AVI, MOV, MKV, WebM).'}), 400

    # Save video to temp file for OpenCV processing
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4',
                                     dir=app.config['UPLOAD_FOLDER'])
    try:
        file.save(tmp.name)
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open video file'}), 400

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        # Sample up to 5 evenly-spaced key frames
        num_samples = min(5, max(1, total_frames))
        if total_frames <= 5:
            sample_indices = list(range(total_frames))
        else:
            sample_indices = [int(i * (total_frames - 1) / (num_samples - 1))
                              for i in range(num_samples)]

        # Import all modules
        from modules.vehicle_counter import detect_vehicles
        from modules.anpr import detect_plates
        from modules.face_recognition_module import analyze_faces
        from modules.people_counter import count_people

        # Aggregated results
        vehicle_totals = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        all_plates = []
        all_faces = []
        total_people = 0
        max_density = 'Low'
        density_order = {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3}

        frame_results = []
        annotated_frames = []

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Encode frame to bytes for module processing
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buf.tobytes()
            timestamp = round(idx / fps, 2) if fps > 0 else 0

            frame_data = {'frame_index': idx, 'timestamp': timestamp}

            # Vehicle detection
            try:
                v_result = detect_vehicles(frame_bytes)
                for vtype, vcount in v_result.get('counts', {}).items():
                    vehicle_totals[vtype] = vehicle_totals.get(vtype, 0) + vcount
                frame_data['vehicles'] = v_result.get('total', 0)
                frame_data['vehicle_annotated'] = v_result.get('annotated_image', '')
            except Exception:
                frame_data['vehicles'] = 0

            # ANPR
            try:
                a_result = detect_plates(frame_bytes)
                for plate in a_result.get('plates', []):
                    plate['timestamp'] = timestamp
                    if plate['text'] not in [p['text'] for p in all_plates]:
                        all_plates.append(plate)
                frame_data['plates'] = a_result.get('total_plates', 0)
            except Exception:
                frame_data['plates'] = 0

            # Face recognition
            try:
                f_result = analyze_faces(frame_bytes)
                for face in f_result.get('faces', []):
                    face['timestamp'] = timestamp
                all_faces.extend(f_result.get('faces', []))
                frame_data['faces'] = f_result.get('total_faces', 0)
                frame_data['face_annotated'] = f_result.get('annotated_image', '')
            except Exception:
                frame_data['faces'] = 0

            # People counting
            try:
                p_result = count_people(frame_bytes)
                frame_people = p_result.get('total_people', 0)
                total_people = max(total_people, frame_people)
                frame_density = p_result.get('density', 'Low')
                if density_order.get(frame_density, 0) > density_order.get(max_density, 0):
                    max_density = frame_density
                frame_data['people'] = frame_people
                frame_data['density'] = frame_density
                frame_data['people_annotated'] = p_result.get('annotated_image', '')
                frame_data['heatmap'] = p_result.get('heatmap_image', '')
            except Exception:
                frame_data['people'] = 0

            frame_results.append(frame_data)

        cap.release()

        return jsonify({
            'video_info': {
                'total_frames': total_frames,
                'fps': round(fps, 1),
                'resolution': f'{width}x{height}',
                'duration': round(duration, 1),
                'frames_analyzed': len(frame_results)
            },
            'vehicle_summary': {
                'counts': vehicle_totals,
                'total': sum(vehicle_totals.values())
            },
            'anpr_summary': {
                'plates': all_plates,
                'total_unique': len(all_plates)
            },
            'face_summary': {
                'faces': all_faces[:20],  # limit response size
                'total_detected': len(all_faces)
            },
            'people_summary': {
                'max_count': total_people,
                'max_density': max_density
            },
            'frame_results': frame_results
        })

    except Exception as e:
        return jsonify({'error': f'Video processing failed: {str(e)}'}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({
        'status': 'online',
        'modules': [
            'vehicle_counter',
            'anpr',
            'face_recognition',
            'people_counter'
        ]
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  AI Video Analytics Dashboard")
    print("  Starting on http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    # Disable reloader to prevent ConnectionResetError/Instability on Windows with large models
    import torch
    gpu_status = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "GPU: NOT FOUND (Using CPU)"
    print(f"  Hardware: {gpu_status}")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
