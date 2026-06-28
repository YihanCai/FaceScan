from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
import cv2
import numpy as np


@dataclass
class Face:
    bbox: tuple  # (x, y, w, h)
    confidence: float = 0.0
    landmarks: Optional[List[tuple]] = None  # 关键点列表 [(x,y), ...]
    key: str = ""


@dataclass
class DetectionResult:
    faces: List[Face] = field(default_factory=list)
    image: Optional[np.ndarray] = None

    @property
    def count(self) -> int:
        return len(self.faces)


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> DetectionResult:
        ...

    def detect_from_file(self, path: str) -> DetectionResult:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"无法读取图片: {path}")
        return self.detect(image)

    def detect_from_video(self, path: str, callback=None):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频: {path}")
        return self._process_video(cap, callback)

    def detect_from_camera(self, camera_id: int = 0, callback=None):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise IOError(f"无法打开摄像头 {camera_id}")
        return self._process_video(cap, callback)

    def _process_video(self, cap: cv2.VideoCapture, callback=None):
        results = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = self.detect(frame)
            result.image = frame
            results.append(result)
            if callback:
                should_break = callback(result)
                if should_break:
                    break
        cap.release()
        cv2.destroyAllWindows()
        return results

    def release(self):
        pass
