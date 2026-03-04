from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
    
    def detect_and_track(self, frame):
        # We use YOLOv8 built-in tracking (uses ByteTrack internally)
        # persist=True ensures object IDs are maintained across frames
        # classes: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck (COCO dataset)
        results = self.model.track(frame, persist=True, classes=[0, 2, 3, 5, 7], verbose=False)
        return results
