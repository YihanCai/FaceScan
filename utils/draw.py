import cv2
import numpy as np
from detector.base import DetectionResult


_COLORS = [
    (0, 255, 0),    # 绿
    (255, 0, 0),    # 蓝
    (0, 0, 255),    # 红
    (255, 255, 0),  # 青
    (255, 0, 255),  # 紫
    (0, 255, 255),  # 黄
]


def draw_faces(image: np.ndarray, result: DetectionResult,
               show_confidence: bool = True, show_landmarks: bool = True,
               landmark_color: tuple = (0, 255, 255)) -> np.ndarray:
    vis = image.copy()
    for i, face in enumerate(result.faces):
        x, y, w, h = face.bbox
        color = _COLORS[i % len(_COLORS)]
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        if show_confidence and face.confidence > 0:
            label = f"{face.confidence:.2f}"
            cv2.putText(vis, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if show_landmarks and face.landmarks:
            for lx, ly in face.landmarks:
                cv2.circle(vis, (lx, ly), 2, landmark_color, -1)
    label = f"Faces: {result.count}"
    cv2.putText(vis, label, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return vis


def draw_fps(image: np.ndarray, fps: float) -> np.ndarray:
    cv2.putText(image, f"FPS: {fps:.1f}", (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return image
