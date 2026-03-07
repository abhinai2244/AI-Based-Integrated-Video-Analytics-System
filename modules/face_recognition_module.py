"""
Facial Recognition & Analysis Module
Uses DeepFace for face detection, age/gender/emotion/race analysis.
"""

import cv2
import numpy as np
from deepface import DeepFace
import base64
import traceback


def analyze_faces(image_bytes):
    """
    Detect faces and analyze attributes (age, gender, emotion, race).
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        dict with keys:
            - faces: list of face analysis dicts
            - annotated_image: base64 encoded annotated image
            - total_faces: count of detected faces
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Could not decode image"}
    
    annotated = img.copy()
    faces = []
    
    try:
        # Run DeepFace analysis with RetinaFace (robust for Indian faces)
        # and Facenet512 for high-resolution feature extraction.
        results = DeepFace.analyze(
            img,
            actions=['age', 'gender', 'emotion', 'race'],
            enforce_detection=True, # Robustly check for faces
            detector_backend='retinaface', # Gold standard for orientation/faces
            model_name='Facenet512',
            silent=True
        )
        
        # Ensure results is a list
        if isinstance(results, dict):
            results = [results]
        
        # Color palette for face boxes
        colors = [
            (0, 255, 200), (255, 100, 200), (100, 200, 255),
            (255, 200, 0), (0, 200, 100), (200, 100, 255)
        ]
        
        for i, result in enumerate(results):
            region = result.get('region', {})
            x = region.get('x', 0)
            y = region.get('y', 0)
            w = region.get('w', 0)
            h = region.get('h', 0)
            
            # Skip if no valid region detected
            if w < 10 or h < 10:
                continue
            
            age = result.get('age', 'N/A')
            gender = result.get('dominant_gender', 'N/A')
            emotion = result.get('dominant_emotion', 'N/A')
            race = result.get('dominant_race', 'N/A')
            
            # Gender confidence
            gender_conf = 0
            if 'gender' in result and isinstance(result['gender'], dict):
                gender_conf = max(result['gender'].values()) / 100.0
            
            # Emotion confidence
            emotion_conf = 0
            if 'emotion' in result and isinstance(result['emotion'], dict):
                emotion_conf = max(result['emotion'].values()) / 100.0
            
            face_data = {
                'id': i + 1,
                'age': age,
                'gender': gender,
                'gender_confidence': round(gender_conf, 3),
                'emotion': emotion,
                'emotion_confidence': round(emotion_conf, 3),
                'race': race,
                'bbox': [x, y, x + w, y + h]
            }
            faces.append(face_data)
            
            # Draw on annotated image
            color = colors[i % len(colors)]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Info label background
            labels = [
                f"Face #{i+1}",
                f"Age: {age}",
                f"Gender: {gender}",
                f"Emotion: {emotion}",
            ]
            
            label_y = y - 10
            for label in reversed(labels):
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x, label_y - th - 5), (x + tw + 5, label_y + 2), color, -1)
                cv2.putText(annotated, label, (x + 2, label_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                label_y -= th + 8
                
    except Exception as e:
        print(f"DeepFace analysis error: {traceback.format_exc()}")
        # Return with empty faces on error
        pass
    
    # Encode annotated image
    _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    annotated_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        'faces': faces,
        'annotated_image': annotated_b64,
        'total_faces': len(faces)
    }
