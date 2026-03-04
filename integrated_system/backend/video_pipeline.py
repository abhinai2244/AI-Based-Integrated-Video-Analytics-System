import cv2
import threading
from detectors.object_detector import ObjectDetector
from detectors.plate_detector import PlateDetector
from ocr.plate_reader import PlateReader
from analytics.vehicle_counter import VehicleCounter
from analytics.people_counter import PeopleCounter
from analytics.face_recognizer import FaceRecognizer
from analytics.gender_classifier import GenderClassifier
from database.db import SessionLocal
from database.models import VehicleEvent, PeopleEvent, ANPRLog, FRSLog

class VideoPipeline:
    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.detector = ObjectDetector(model_path="yolov8n.pt")
        self.plate_detector = PlateDetector()
        self.plate_reader = PlateReader()
        
        self.vehicle_counter = VehicleCounter(self.width, self.height)
        self.people_counter = PeopleCounter(self.width, self.height)
        
        self.face_recognizer = FaceRecognizer(
            "../models/face_gender/opencv_face_detector.pbtxt",
            "../models/face_gender/res10_300x300_ssd_iter_140000.caffemodel"
        )
        self.gender_classifier = GenderClassifier(
            "../models/face_gender/deploy.prototxt",
            "../models/face_gender/gender_net.caffemodel"
        )
        
        self.current_frame = None
        self.running = False

    def process_frame(self, frame):
        # 1. Detect and track all objects using YOLOv8
        results = self.detector.detect_and_track(frame)
        
        if not results or not results[0].boxes or results[0].boxes.id is None:
            return frame
            
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        
        # 2. Iterate detections and route to analytics modules
        for box, track_id, cls in zip(boxes, track_ids, classes):
            track_id = int(track_id)
            cls = int(cls)
            
            # People (class 0)
            if cls == 0:
                result = self.people_counter.process_person(box, track_id)
                
                # Check Face and Gender
                x1, y1, x2, y2 = map(int, box)
                crop = frame[max(0, y1-10):min(self.height, y2+10), max(0, x1-10):min(self.width, x2+10)]
                
                gender = "Unknown"
                recognized_name = "Unknown"
                if crop.size > 0:
                    faces = self.face_recognizer.detect_face(crop)
                    if faces:
                        fx1, fy1, fx2, fy2 = faces[0]
                        face_crop = crop[max(0, fy1):min(crop.shape[0], fy2), max(0, fx1):min(crop.shape[1], fx2)]
                        if face_crop.size > 0:
                            gender = self.gender_classifier.classify_gender(face_crop)
                            recognized_name = self.face_recognizer.recognize(face_crop)
                            
                if result["event"]:
                    # DB insert logic
                    with SessionLocal() as db:
                        db.add(PeopleEvent(track_id=track_id, direction=result["event"], gender=gender))
                        if recognized_name != "Unknown":
                            db.add(FRSLog(track_id=track_id, recognized_name=recognized_name, confidence=1.0))
                        db.commit()
            
            # Vehicles (class 2: car, 3: motorcycle, 5: bus, 7: truck)
            elif cls in [2, 3, 5, 7]:
                # Count
                result = self.vehicle_counter.count_obj(box, track_id, cls)
                if result["event"]:
                    with SessionLocal() as db:
                        cls_name = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}[cls]
                        db.add(VehicleEvent(track_id=track_id, vehicle_type=cls_name, direction=result["event"]))
                        db.commit()
                        
                # ANPR: Extract crop and pass to PlateDetector
                x1, y1, x2, y2 = map(int, box)
                # Expand crop slightly to ensure plate is fully visible
                crop = frame[max(0, y1-10):min(self.height, y2+10), max(0, x1-10):min(self.width, x2+10)]
                
                if crop.size > 0:
                    plate_box, conf = self.plate_detector.detect(crop)
                    if plate_box:
                        px1, py1, px2, py2 = plate_box
                        plate_crop = crop[py1:py2, px1:px2]
                        
                        plate_text, score = self.plate_reader.read_plate(plate_crop)
                        if plate_text and score > 0.4:  # Threshold
                            with SessionLocal() as db:
                                # Quick check to not spam DB for same plate
                                existing = db.query(ANPRLog).filter(ANPRLog.track_id == track_id).first()
                                if not existing or existing.confidence < score:
                                    if existing:
                                        existing.plate_text = plate_text
                                        existing.confidence = score
                                    else:
                                        db.add(ANPRLog(track_id=track_id, plate_text=plate_text, confidence=score))
                                    db.commit()

        # Draw annotations (Basic wrapper for YOLOv8 built-in plotter)
        annotated_frame = results[0].plot()

        # Overlay Counters
        cv2.putText(annotated_frame, f"Vehicles Out (North): {self.vehicle_counter.up_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Vehicles In (South): {self.vehicle_counter.down_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(annotated_frame, f"People Enter: {self.people_counter.totalDown}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(annotated_frame, f"People Exit: {self.people_counter.totalUp}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return annotated_frame

    def start(self):
        self.running = True
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                # Loop video for demo purposes
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            self.current_frame = self.process_frame(frame)
            
    def stop(self):
        self.running = False
        self.cap.release()
