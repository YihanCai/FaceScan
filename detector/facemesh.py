import os
import cv2
import numpy as np
from .base import BaseDetector, DetectionResult, Face

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models",
                           "face_landmarker.task")

LEFT_EYE_CENTER = 468
RIGHT_EYE_CENTER = 473


class FaceMeshDetector(BaseDetector):
    def __init__(self, model_path: str = "", min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5, num_faces: int = 10):
        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError("请先安装 mediapipe: pip install mediapipe")
        self.mp = mp
        model = model_path or os.path.abspath(_MODEL_PATH)
        if not os.path.isfile(model):
            raise FileNotFoundError(
                f"模型文件不存在: {model}\n"
                f"请手动下载: https://storage.googleapis.com/mediapipe-models/"
                f"face_landmarker/face_landmarker/float16/1/"
                f"face_landmarker.task"
            )
        from mediapipe.tasks.python.vision import (
            FaceLandmarker, FaceLandmarkerOptions, RunningMode
        )
        from mediapipe.tasks.python import BaseOptions
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model),
            running_mode=RunningMode.IMAGE,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.detector = FaceLandmarker.create_from_options(options)

    def detect(self, image: np.ndarray) -> DetectionResult:
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        results = self.detector.detect(mp_image)
        result = DetectionResult(image=image)
        h, w, _ = image.shape
        if results.face_landmarks:
            for landmarks in results.face_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                face = Face(bbox=(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)),
                            confidence=1.0, landmarks=pts)
                result.faces.append(face)
        return result

    def release(self):
        self.detector.close()
