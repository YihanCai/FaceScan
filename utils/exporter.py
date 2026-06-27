import json
import csv
import os
from typing import List
from detector.base import DetectionResult, Face


def _face_to_dict(face: Face) -> dict:
    d = {
        "bbox": {
            "x": face.bbox[0],
            "y": face.bbox[1],
            "w": face.bbox[2],
            "h": face.bbox[3],
        },
        "confidence": face.confidence,
    }
    if face.landmarks:
        d["landmarks"] = [{"x": p[0], "y": p[1]} for p in face.landmarks]
    return d


def export_json(results: List[DetectionResult], output_path: str):
    data = []
    for i, r in enumerate(results):
        faces_data = [_face_to_dict(f) for f in r.faces]
        data.append({
            "frame": i,
            "face_count": r.count,
            "faces": faces_data,
        })
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def export_csv(results: List[DetectionResult], output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "face_index", "x", "y", "w", "h", "confidence"])
        for i, r in enumerate(results):
            for j, face in enumerate(r.faces):
                writer.writerow([
                    i, j,
                    face.bbox[0], face.bbox[1],
                    face.bbox[2], face.bbox[3],
                    f"{face.confidence:.4f}",
                ])
