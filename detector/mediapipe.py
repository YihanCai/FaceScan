import cv2
import numpy as np
from .base import BaseDetector, DetectionResult, Face


class MediaPipeDetector(BaseDetector):
    def __init__(self, min_detection_confidence: float = 0.5):
        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError("请先安装 mediapipe: pip install mediapipe")
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )

    def detect(self, image: np.ndarray) -> DetectionResult:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        result = DetectionResult(image=image)
        if results.detections:
            h, w, _ = image.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                confidence = detection.score[0]
                face = Face(bbox=(x, y, bw, bh), confidence=confidence)
                keypoints = detection.location_data.relative_keypoints
                face.landmarks = [(int(kp.x * w), int(kp.y * h)) for kp in keypoints]
                result.faces.append(face)
        return result

    def release(self):
        self.detector.close()
