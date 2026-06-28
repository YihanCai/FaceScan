import os
import cv2
import glob
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from detector import HaarDetector, MediaPipeDetector, FaceMeshDetector
from utils import draw_faces, export_json, export_csv
from utils.align import align_face


class FaceScanGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("FaceScan - 人脸检测系统")
        self.root.geometry("1200x750")
        self.root.minsize(900, 600)

        self.detector = HaarDetector()
        self.running = False
        self.cap = None
        self.current_result = None
        self.current_image = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI 构建 ──────────────────────────────────────

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # 左：控制面板
        ctrl = ttk.LabelFrame(main, text="控制面板", width=280)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        ctrl.pack_propagate(False)

        self._build_mode_section(ctrl)
        self._build_source_section(ctrl)
        self._build_detector_section(ctrl)
        self._build_options_section(ctrl)
        self._build_action_buttons(ctrl)
        self._build_export_section(ctrl)

        # 右：图像显示 + 结果
        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_display_area(right)
        self._build_result_log(right)

    def _build_mode_section(self, parent):
        f = ttk.LabelFrame(parent, text="运行模式", padding=4)
        f.pack(fill=tk.X, pady=(0, 6))
        self.mode_var = tk.StringVar(value="image")
        for text, val in [("图片", "image"), ("摄像头", "camera"),
                          ("视频", "video"), ("批量", "batch")]:
            ttk.Radiobutton(f, text=text, variable=self.mode_var,
                            value=val, command=self._on_mode_change).pack(
                anchor=tk.W, padx=4, pady=1)

    def _build_source_section(self, parent):
        f = ttk.LabelFrame(parent, text="输入源", padding=4)
        f.pack(fill=tk.X, pady=(0, 6))
        self.source_var = tk.StringVar()
        entry = ttk.Entry(f, textvariable=self.source_var)
        entry.pack(fill=tk.X, padx=2, pady=2)
        btn_frame = ttk.Frame(f)
        btn_frame.pack(fill=tk.X)
        self.browse_btn = ttk.Button(btn_frame, text="浏览...",
                                     command=self._browse)
        self.browse_btn.pack(side=tk.LEFT, padx=2, pady=2)
        self.camera_id_label = ttk.Label(btn_frame, text="")
        self.camera_id_label.pack(side=tk.LEFT, padx=4)
        self._on_mode_change()

    def _build_detector_section(self, parent):
        f = ttk.LabelFrame(parent, text="检测器", padding=4)
        f.pack(fill=tk.X, pady=(0, 6))
        self.detector_var = tk.StringVar(value="haar")
        combo = ttk.Combobox(f, textvariable=self.detector_var,
                             values=["haar", "mediapipe", "facemesh"],
                             state="readonly")
        combo.pack(fill=tk.X, padx=2, pady=2)
        combo.bind("<<ComboboxSelected>>", self._on_detector_change)

    def _build_options_section(self, parent):
        f = ttk.LabelFrame(parent, text="选项", padding=4)
        f.pack(fill=tk.X, pady=(0, 6))
        self.save_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="保存结果", variable=self.save_var).pack(
            anchor=tk.W, padx=4, pady=1)
        self.align_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="人脸对齐（需 facemesh）",
                        variable=self.align_var).pack(
            anchor=tk.W, padx=4, pady=1)

    def _build_action_buttons(self, parent):
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=(0, 6))
        self.start_btn = ttk.Button(f, text="▶ 开始", command=self._start)
        self.start_btn.pack(side=tk.LEFT, fill=tk.X, expand=True,
                            padx=2, pady=2)
        self.stop_btn = ttk.Button(f, text="⏹ 停止", command=self._stop,
                                   state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, fill=tk.X, expand=True,
                           padx=2, pady=2)

    def _build_export_section(self, parent):
        f = ttk.LabelFrame(parent, text="导出", padding=4)
        f.pack(fill=tk.X, pady=(0, 6))
        self.export_var = tk.StringVar(value="")
        ttk.Button(f, text="导出 JSON", command=lambda: self._export("json")
                   ).pack(fill=tk.X, padx=2, pady=2)
        ttk.Button(f, text="导出 CSV", command=lambda: self._export("csv")
                   ).pack(fill=tk.X, padx=2, pady=2)

    def _build_display_area(self, parent):
        f = ttk.LabelFrame(parent, text="检测结果", padding=4)
        f.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Label(f, text="选择图片或启动摄像头开始检测",
                               bg="#1e1e1e", fg="#888",
                               font=("Arial", 14))
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def _build_result_log(self, parent):
        f = ttk.LabelFrame(parent, text="日志", padding=4, height=150)
        f.pack(fill=tk.X, pady=(4, 0))
        self.log_text = tk.Text(f, height=6, state=tk.DISABLED,
                                font=("Consolas", 10))
        scroll = ttk.Scrollbar(f, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # ── 模式切换 ──────────────────────────────────────

    def _on_mode_change(self):
        mode = self.mode_var.get()
        if mode in ("image", "video"):
            self.browse_btn.configure(text="浏览文件...")
            self.camera_id_label.configure(text="")
            self.source_var.set("")
        elif mode == "batch":
            self.browse_btn.configure(text="浏览目录...")
            self.camera_id_label.configure(text="")
            self.source_var.set("")
        else:
            self.browse_btn.configure(text="")
            self.camera_id_label.configure(text="摄像头 ID (默认 0)")
            self.source_var.set("0")

    def _on_detector_change(self, event=None):
        name = self.detector_var.get()
        try:
            if name == "haar":
                self.detector = HaarDetector()
            elif name == "mediapipe":
                self.detector = MediaPipeDetector()
            elif name == "facemesh":
                self.detector = FaceMeshDetector()
            self._log(f"检测器已切换: {name}")
            self._set_controls(True)
        except Exception as e:
            self._log(f"切换失败: {e}")
            self.detector_var.set("haar")
            self.detector = HaarDetector()

    # ── 浏览 ──────────────────────────────────────────

    def _browse(self):
        mode = self.mode_var.get()
        if mode == "image":
            path = filedialog.askopenfilename(
                title="选择图片",
                filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")])
        elif mode == "video":
            path = filedialog.askopenfilename(
                title="选择视频",
                filetypes=[("视频", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")])
        elif mode == "batch":
            path = filedialog.askdirectory(title="选择图片目录")
        else:
            return
        if path:
            self.source_var.set(path)

    # ── 启动 / 停止 ────────────────────────────────────

    def _set_controls(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        for child in self.root.winfo_children():
            self._set_state(child, state if enabled else tk.DISABLED)

    def _set_state(self, widget, state):
        try:
            widget.configure(state=state)
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._set_state(child, state)

    def _start(self):
        self.running = True
        mode = self.mode_var.get()
        self._on_detector_change()
        self.stop_btn.configure(state=tk.NORMAL)
        self.start_btn.configure(state=tk.DISABLED)

        if mode == "image":
            threading.Thread(target=self._run_image, daemon=True).start()
        elif mode == "camera":
            threading.Thread(target=self._run_camera, daemon=True).start()
        elif mode == "video":
            threading.Thread(target=self._run_video, daemon=True).start()
        elif mode == "batch":
            threading.Thread(target=self._run_batch, daemon=True).start()

    def _stop(self):
        self.running = False
        self.stop_btn.configure(state=tk.DISABLED)
        self.start_btn.configure(state=tk.NORMAL)

    # ── 图片模式 ──────────────────────────────────────

    def _run_image(self):
        path = self.source_var.get()
        if not path or not os.path.isfile(path):
            self._log(f"文件不存在: {path}")
            self._stop()
            return
        try:
            result = self.detector.detect_from_file(path)
            result.source = path
            self.current_result = [result]
            self.current_image = result.image.copy()
            self._show_image(result.image, result)
            if self.save_var.get():
                self._save_result(result, path)
            self._log(f"检测完毕: {result.count} 张人脸")
        except Exception as e:
            self._log(f"检测失败: {e}")
        self._stop()

    # ── 摄像头模式 ─────────────────────────────────────

    def _run_camera(self):
        src = self.source_var.get() or "0"
        cam_id = int(src) if src.isdigit() else 0
        self.cap = cv2.VideoCapture(cam_id)
        if not self.cap.isOpened():
            self._log(f"无法打开摄像头 {cam_id}")
            self._stop()
            return
        self._log(f"摄像头 {cam_id} 已开启")
        self.current_result = []
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            result = self.detector.detect(frame)
            result.image = frame
            self.current_result.append(result)
            self._show_image(frame, result)
        self.cap.release()
        self.cap = None
        self._stop()

    # ── 视频模式 ──────────────────────────────────────

    def _run_video(self):
        path = self.source_var.get()
        if not path or not os.path.isfile(path):
            self._log(f"文件不存在: {path}")
            self._stop()
            return
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self._log(f"无法打开视频: {path}")
            self._stop()
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.save_var.get():
            base = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join("output", f"{base}_result.mp4")
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (w, h))
        self.current_result = []
        frame_idx = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            result = self.detector.detect(frame)
            result.image = frame
            self.current_result.append(result)
            vis = draw_faces(frame.copy(), result)
            self._show_image(vis, None)
            if self.save_var.get():
                writer.write(vis)
            frame_idx += 1
        self.cap.release()
        if self.save_var.get():
            writer.release()
            self._log(f"视频已保存: {out_path}")
        self._log(f"处理 {len(self.current_result)} 帧")
        self.cap = None
        self._stop()

    # ── 批量模式 ──────────────────────────────────────

    def _run_batch(self):
        path = self.source_var.get()
        if not path or not os.path.isdir(path):
            self._log(f"目录不存在: {path}")
            self._stop()
            return
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        files = sorted(set(
            f for ext in exts
            for f in glob.glob(os.path.join(path, ext)) +
            glob.glob(os.path.join(path, ext.upper()))
        ))
        if not files:
            self._log(f"目录中没有图片: {path}")
            self._stop()
            return
        self._log(f"找到 {len(files)} 张图片")
        all_results = []
        for idx, fp in enumerate(files):
            if not self.running:
                break
            try:
                result = self.detector.detect_from_file(fp)
                result.source = fp
                all_results.append(result)
                self.current_result = all_results
                self.current_image = result.image.copy()
                self._show_image(result.image, result)
                if self.save_var.get():
                    self._save_result(result, fp)
                self._log(f"[{idx + 1}/{len(files)}] {os.path.basename(fp)}: "
                          f"{result.count} 张人脸")
            except Exception as e:
                self._log(f"[{idx + 1}/{len(files)}] {os.path.basename(fp)}: 失败 - {e}")
        self._log(f"批量处理完成，共 {len(all_results)} 张图片")
        self._stop()

    # ── 图像显示 ──────────────────────────────────────

    def _show_image(self, image, result=None):
        display = image.copy()
        if result is not None:
            display = draw_faces(display, result)
        # 缩放以适应显示区域
        max_w, max_h = 800, 500
        h_orig, w_orig = display.shape[:2]
        scale = min(max_w / w_orig, max_h / h_orig, 1.0)
        if scale < 1:
            new_w, new_h = int(w_orig * scale), int(h_orig * scale)
            display = cv2.resize(display, (new_w, new_h))
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.configure(image=self.tk_img, text="")

    # ── 保存 / 导出 ──────────────────────────────────

    def _save_result(self, result, source_path):
        base = os.path.splitext(os.path.basename(source_path))[0]
        output = "output"
        os.makedirs(output, exist_ok=True)
        vis = result.image.copy()
        vis = draw_faces(vis, result, show_landmarks=True)
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(vis)
        img.save(os.path.join(output, f"{base}_result.jpg"))

    def _export(self, fmt: str):
        if not self.current_result:
            self._log("没有可导出的结果")
            return
        path = os.path.join("output", f"results.{fmt}")
        os.makedirs("output", exist_ok=True)
        if fmt == "json":
            export_json(self.current_result, path)
        else:
            export_csv(self.current_result, path)
        self._log(f"结果已导出: {path}")

    # ── 日志 ──────────────────────────────────────────

    def _log(self, msg: str):
        self.root.after(0, self._do_log, msg)

    def _do_log(self, msg: str):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    # ── 关闭 ──────────────────────────────────────────

    def _on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        try:
            self.detector.release()
        except Exception:
            pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()
