import cv2
import numpy as np
from typing import List, Tuple

# MediaPipe FaceMesh indices
LEFT_EYE_CENTER = 468
RIGHT_EYE_CENTER = 473


def get_eye_centers(landmarks: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    if len(landmarks) <= max(LEFT_EYE_CENTER, RIGHT_EYE_CENTER):
        return None, None
    return landmarks[LEFT_EYE_CENTER], landmarks[RIGHT_EYE_CENTER]


def align_face(image: np.ndarray, landmarks: List[Tuple[int, int]],
               output_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    left_eye, right_eye = get_eye_centers(landmarks)
    if left_eye is None or right_eye is None:
        raise ValueError("关键点数量不足，无法进行对齐")

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                   (left_eye[1] + right_eye[1]) // 2)

    M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                             flags=cv2.INTER_CUBIC)

    # 根据眼睛间距估算裁剪框大小
    eye_dist = np.sqrt(dx ** 2 + dy ** 2)
    crop_size = int(eye_dist * 3.0)
    cx, cy = eyes_center
    x1 = max(0, cx - crop_size // 2)
    y1 = max(0, cy - int(crop_size * 0.6))
    x2 = min(rotated.shape[1], x1 + crop_size)
    y2 = min(rotated.shape[0], y1 + crop_size)

    cropped = rotated[y1:y2, x1:x2]
    if cropped.size == 0:
        raise ValueError("裁剪区域无效")

    return cv2.resize(cropped, output_size, interpolation=cv2.INTER_CUBIC)


def crop_face(image: np.ndarray, bbox: Tuple[int, int, int, int],
              margin: float = 0.3, output_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    x, y, w, h = bbox
    h_img, w_img = image.shape[:2]

    margin_x = int(w * margin)
    margin_y = int(h * margin)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w_img, x + w + margin_x)
    y2 = min(h_img, y + h + margin_y)

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return np.zeros((*output_size, 3), dtype=np.uint8)

    return cv2.resize(cropped, output_size, interpolation=cv2.INTER_CUBIC)
