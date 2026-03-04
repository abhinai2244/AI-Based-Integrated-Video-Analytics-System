import torch
import cv2

class PlateDetector:
    def __init__(self, weights_path="../models/license_plate_detector.pt"):
        # Load custom YOLOv5 model for license plates
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)
        self.model.conf = 0.5

    def detect(self, img):
        if img is None or img.size == 0:
            return None, 0.0
        
        # Convert BGR to RGB if needed by model (yolov5 handles BGR but RGB is safer)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model([img_rgb])
        detections = results.xyxy[0]
        
        best_plate = None
        best_conf = 0.0
        
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if conf > best_conf:
                best_conf = float(conf)
                best_plate = (int(x1), int(y1), int(x2), int(y2))
                
        return best_plate, best_conf
