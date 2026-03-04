import cv2
import numpy as np
import easyocr
import re
import logging

class PlateReader:
    def __init__(self):
        # Initialize EasyOCR
        try:
            self.reader = easyocr.Reader(['en'], gpu=True)
        except Exception as e:
            logging.warning(f"Failed to load EasyOCR with GPU, falling back to CPU: {e}")
            self.reader = easyocr.Reader(['en'], gpu=False)
        
    def clean_image(self, img):
        # Resize, convert to grayscale and threshold
        img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        kernel = np.ones((1, 1), np.uint8)
        gray = cv2.dilate(gray, kernel, iterations=1)
        gray = cv2.erode(gray, kernel, iterations=1)
        return gray

    def read_plate(self, plate_img):
        if plate_img is None or plate_img.size == 0:
            return "", 0.0
            
        try:
            cleaned_img = self.clean_image(plate_img)
            ocr_result = self.reader.readtext(cleaned_img)
            
            plate_text = ""
            best_score = 0.0
            
            for result in ocr_result:
                bbox, text, score = result
                # Filter text: only alphanumeric
                text = re.sub(r'\\W+', '', text).upper()
                if score > best_score and len(text) >= 4:
                    best_score = float(score)
                    plate_text = text
                    
            return plate_text, best_score
        except Exception as e:
            logging.error(f"Error in OCR: {e}")
            return "", 0.0
