import cv2
import numpy as np
from .base import BaseDetector, DetectionResult, Face


class HaarDetector(BaseDetector):
    def __init__(self, cascade_path: str = "", scale_factor: float = 1.1,
                 min_neighbors: int = 5, min_size=(30, 30)):
        if cascade_path:
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, image: np.ndarray) -> DetectionResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_rects = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        result = DetectionResult(image=image)
        for (x, y, w, h) in faces_rects:
            result.faces.append(Face(bbox=(x, y, w, h), confidence=1.0))
        return result
