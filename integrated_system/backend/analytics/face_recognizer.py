import cv2
import numpy as np
import os

class FaceRecognizer:
    def __init__(self, detector_txt, detector_bin):
        self.net = cv2.dnn.readNetFromCaffe(detector_txt, detector_bin)
        self.watchlist = {}  # {name: embedding}
        
    def detect_face(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                faces.append((startX, startY, endX, endY))
        return faces

    def recognize(self, face_crop):
        return "Unknown" # Simple placeholder until watchlist is added
