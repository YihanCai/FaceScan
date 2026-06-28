import argparse
import glob
import os
import sys
import time
import cv2
import numpy as np
from detector import HaarDetector, MediaPipeDetector, FaceMeshDetector
from utils import draw_faces, draw_fps, export_json, export_csv
from utils.align import align_face, crop_face


def imwrite(path: str, img) -> bool:
    _, buf = cv2.imencode(os.path.splitext(path)[1], img)
    return buf.tofile(path)


def get_detector(name: str):
    if name == "haar":
        return HaarDetector()
    elif name == "mediapipe":
        return MediaPipeDetector()
    elif name == "facemesh":
        return FaceMeshDetector()
    else:
        raise ValueError(f"不支持的检测器: {name}，可选: haar, mediapipe, facemesh")


def process_image(detector, source: str, output: str, save: bool, display: bool,
                  align: bool = False):
    result = detector.detect_from_file(source)
    result.source = source
    print(f"检测到 {result.count} 张人脸")
    for i, face in enumerate(result.faces):
        x, y, w, h = face.bbox
        conf = face.confidence
        kp = f", 关键点: {len(face.landmarks)}个" if face.landmarks else ""
        print(f"  人脸 {i + 1}: ({x}, {y}, {w}, {h}) 置信度: {conf:.4f}{kp}")
        if align and face.landmarks and len(face.landmarks) > 470:
            aligned = align_face(result.image, face.landmarks)
            base = os.path.splitext(os.path.basename(source))[0]
            align_path = os.path.join(output, f"{base}_face{i}_aligned.jpg")
            imwrite(align_path, aligned)
            print(f"    对齐结果: {align_path}")
    if save or display:
        vis = draw_faces(result.image, result, show_landmarks=True)
        base = os.path.splitext(os.path.basename(source))[0]
        if save:
            out_path = os.path.join(output, f"{base}_result.jpg")
            imwrite(out_path, vis)
            print(f"结果已保存: {out_path}")
        if display:
            cv2.imshow("FaceScan", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return [result]


def process_batch(detector, source_dir: str, output: str, save: bool, display: bool,
                  align: bool = False, export_format: str = None):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(source_dir, ext)))
        files.extend(glob.glob(os.path.join(source_dir, ext.upper())))
    files = sorted(set(files))
    if not files:
        print(f"目录中没有找到图片: {source_dir}")
        sys.exit(1)
    print(f"找到 {len(files)} 张图片，开始批量处理...")
    all_results = []
    for idx, path in enumerate(files):
        print(f"[{idx + 1}/{len(files)}] {os.path.basename(path)}")
        results = process_image(detector, path, output, save, display, align)
        all_results.extend(results)
    print(f"批量处理完成，共 {len(all_results)} 张图片")
    if export_format and all_results:
        ext = export_format
        out_name = f"batch_results.{ext}"
        out_path = os.path.join(output, out_name)
        if ext == "json":
            export_json(all_results, out_path)
        else:
            export_csv(all_results, out_path)
        print(f"批量结果已导出: {out_path}")


def process_video_or_camera(detector, source, output: str, save: bool, display: bool):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        src_name = f"camera_{source}"
    elif os.path.isfile(source):
        cap = cv2.VideoCapture(source)
        src_name = os.path.splitext(os.path.basename(source))[0]
    else:
        print(f"无效的视频源: {source}")
        sys.exit(1)
    if not cap.isOpened():
        print(f"无法打开视频源: {source}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = os.path.join(output, f"{src_name}_result.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))
    all_results = []
    frame_idx = 0
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = detector.detect(frame)
        result.image = frame
        all_results.append(result)
        vis = draw_faces(frame, result)
        elapsed = time.time() - start_time
        current_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
        vis = draw_fps(vis, current_fps)
        if save:
            writer.write(vis)
        if display:
            cv2.imshow("FaceScan", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_idx += 1
    cap.release()
    if save:
        writer.release()
        print(f"视频已保存: {out_path}")
    cv2.destroyAllWindows()
    if all_results:
        total_faces = sum(r.count for r in all_results)
        print(f"处理 {len(all_results)} 帧，共检测到 {total_faces} 张人脸")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="FaceScan - 人脸检测系统")
    parser.add_argument("--mode", choices=["image", "video", "camera", "batch", "gui"],
                        default="image", help="运行模式")
    parser.add_argument("--source", "-s", default="",
                        help="输入源：图片路径 / 视频路径 / 摄像头ID / 图片目录")
    parser.add_argument("--detector", "-d", choices=["haar", "mediapipe", "facemesh"],
                        default="haar", help="检测器类型")
    parser.add_argument("--output", "-o", default="output",
                        help="输出目录")
    parser.add_argument("--save", action="store_true",
                        help="保存检测结果")
    parser.add_argument("--display", action="store_true",
                        help="显示检测结果窗口")
    parser.add_argument("--export", choices=["json", "csv"],
                        help="导出检测结果格式")
    parser.add_argument("--align", action="store_true",
                        help="人脸对齐（需 facemesh 检测器）")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.mode == "gui":
        from gui import FaceScanGUI
        FaceScanGUI().run()
        return

    detector = get_detector(args.detector)

    try:
        if args.mode == "image":
            src = args.source
            if not src:
                src = input("请输入图片路径: ").strip()
            if not os.path.isfile(src):
                print(f"文件不存在: {src}")
                sys.exit(1)
            process_image(detector, src, args.output, args.save, args.display, args.align)
        elif args.mode == "batch":
            src = args.source
            if not src:
                src = input("请输入图片目录路径: ").strip()
            if not os.path.isdir(src):
                print(f"目录不存在: {src}")
                sys.exit(1)
            process_batch(detector, src, args.output, args.save, args.display,
                          args.align, args.export)
        elif args.mode in ("video", "camera"):
            src = args.source
            if not src:
                if args.mode == "video":
                    src = input("请输入视频路径: ").strip()
                else:
                    src = "0"
            display = args.display or args.mode == "camera"
            results = process_video_or_camera(
                detector, src, args.output, args.save, display
            )
            if args.export and results:
                ext = args.export
                out_name = f"results.{ext}"
                out_path = os.path.join(args.output, out_name)
                if ext == "json":
                    export_json(results, out_path)
                else:
                    export_csv(results, out_path)
                print(f"结果已导出: {out_path}")
    finally:
        try:
            detector.release()
        except Exception:
            pass


if __name__ == "__main__":
    main()
