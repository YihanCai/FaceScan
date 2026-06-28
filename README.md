# FaceScan

A face detection system based on computer vision and deep learning. Supports detecting faces from images, video files, and real-time camera streams.

## Features

- **Image Detection** — detect faces in JPG/PNG/BMP images
- **Video Detection** — process video files frame by frame
- **Real-time Camera** — live face detection with webcam
- **Multi-face Support** — detects multiple faces simultaneously
- **Face Landmarks** — 478-point facial landmark detection
- **Face Alignment** — auto-rotate and crop faces
- **Batch Processing** — process entire folders at once
- **Result Export** — save results as JSON or CSV
- **Desktop GUI** — user-friendly Tkinter interface
- **Dual Detectors** — Haar Cascade, MediaPipe, FaceMesh

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Detect faces in an image
python main.py --mode image --source data/lena.jpg --display

# Real-time camera detection
python main.py --mode camera

# Process a video file
python main.py --mode video --source video.mp4 --save

# Export detection results
python main.py --mode video --source video.mp4 --export json

# Launch desktop GUI
python main.py --mode gui
```

Advanced usage:

```bash
# Use facemesh detector for landmarks
python main.py --mode image --source data/lena.jpg --detector facemesh --display

## Project Structure

```
FaceScan/
├── main.py              # CLI entry point
├── detector/            # Detection module
│   ├── base.py          # Abstract base class & data models
│   ├── haar.py          # Haar Cascade detector
│   ├── mediapipe.py     # MediaPipe face detector
│   └── facemesh.py      # FaceMesh 478-point landmark detector
├── utils/               # Utilities
│   ├── draw.py          # Bbox / landmarks / FPS drawing
│   ├── align.py         # Face alignment & cropping
│   └── exporter.py      # JSON / CSV export
├── gui/                 # Desktop GUI
│   ├── __init__.py
│   └── main_window.py   # Tkinter main window
├── models/              # Pretrained model files
├── data/                # Test images / videos
├── output/              # Detection results
├── requirements.txt
└── README.md
```

## Development Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Done | Basic detection (image, video, camera) |
| Phase 2 | ✅ Done | Landmarks, alignment, batch processing |
| Phase 3 | ✅ Done | Desktop GUI (Tkinter) |
| Phase 4 | ❌ | Future enhancements |

## License

MIT
