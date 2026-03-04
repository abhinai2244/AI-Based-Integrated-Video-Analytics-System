import cv2
import numpy as np

class GenderClassifier:
    def __init__(self, model_txt, model_bin):
        self.net = cv2.dnn.readNetFromCaffe(model_txt, model_bin)
        self.gender_list = ['Male', 'Female']
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    def classify_gender(self, face_crop):
        if face_crop is None or face_crop.size == 0:
            return "Unknown"
            
        blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
        self.net.setInput(blob)
        gender_preds = self.net.forward()
        gender = self.gender_list[gender_preds[0].argmax()]
        return gender
