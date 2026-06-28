# FaceScan

A face detection system based on computer vision and deep learning. Supports detecting faces from images, video files, and real-time camera streams.

## Features

- **Image Detection** — detect faces in JPG/PNG/BMP images
- **Video Detection** — process video files frame by frame
- **Real-time Camera** — live face detection with webcam
- **Multi-face Support** — detects multiple faces simultaneously
- **Result Export** — save results as JSON or CSV
- **Dual Detectors** — Haar Cascade (default, zero dependencies) and MediaPipe (higher accuracy)

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

# Use MediaPipe detector (requires: pip install mediapipe)
python main.py --mode image --source data/lena.jpg --detector mediapipe
```

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
| Phase 3 | ⏳ Next | Desktop GUI (Tkinter/PyQt) |

## License

MIT
