# AGENTS.md — Hummingbird Detector

A Raspberry Pi camera app that detects hummingbirds using OpenCV background subtraction + YOLO.

## Entrypoints

- **`app.py`** — main application (run with `python app.py`). Flask server on `0.0.0.0:5000`.
- **`focus_tuner.py`** — standalone focus tuning tool for Pi Camera Module 3 (port 5001).
- **`stream.py`** — minimal MJPEG test script (has runtime error: references undefined `FOCUS_LOCK_DIOPTERS`).

## Setup

**Raspberry Pi (picamera2):**
```
python3 -m venv --system-site-packages venv   # --system-site-packages required for picamera2
source venv/bin/activate
pip install flask flask-sock ultralytics
```
System deps (not pip): `sudo apt install python3-picamera2 python3-libcamera python3-opencv`

**Dev mode (Mac / no Pi camera):**
```
pip install flask flask-sock opencv-python ultralytics
python app.py --no-camera
```

## Commands

- `python app.py` — run detector (Pi camera)
- `python app.py --no-camera` — run with webcam fallback (dev)
- `python focus_tuner.py` — focus tuning tool (port 5001)
- `python focus_tuner.py --no-camera` — focus tuner with webcam (port 5001, slider has no effect on webcam)

## Key config (in `app.py`)

| Constant | Default | Notes |
|---|---|---|
| `YOLO_MODEL_PATH` | `yolov8n.pt` | Swap for custom hummingbird model |
| `YOLO_CLASSES` | `None` (all) | Set `[14]` for COCO "bird" only |
| `FOCUS_LOCK_DIOPTERS` | `0.25` | Fixed focus (~4ft); use `focus_tuner.py` to find |
| `STREAM_W/STREAM_H` | `1280x720` | Camera resolution |
| `BGS_THRESHOLD` | `40` | Motion sensitivity (lower = more) |
| `MOTION_MIN_AREA` | `800` | px² — filters noise |

## Architecture

- **Online (WS client connected)**: BGS → YOLO (throttled 0.5s) on motion frames → annotated MJPEG stream + JSON over WebSocket.
- **Offline (no client)**: BGS → save motion clips as MP4 → queue for offline YOLO worker thread → results logged to `detections.jsonl`.
- YOLO is optional — graceful fallback if `ultralytics` not installed.
- Routes: `/` (HTML viewer), `/video_feed` (MJPEG), `/ws` (WebSocket JSON), `/status` (JSON).
- `clips/` and `detections.jsonl` are gitignored.

## Focus tuning

Run `focus_tuner.py` on the Pi (port 5001). Use the web UI slider to find optimal `LensPosition`. Click "Lock & Copy to app.py" to get the value for `FOCUS_LOCK_DIOPTERS` in `app.py`.
