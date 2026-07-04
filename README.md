# Hummingbird Detector

## Setup (Raspberry Pi)

```bash
# System packages
sudo apt update
sudo apt install -y python3-picamera2 python3-libcamera python3-opencv

# Python packages (use venv)
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install flask flask-sock ultralytics

# Run
python app.py
```

Then open http://hummingcam.local:5000 in your browser.

## Dev mode (Mac / no Pi camera)

```bash
pip install flask flask-sock opencv-python ultralytics
python app.py --no-camera
```

## Architecture

```
Camera (picamera2 or webcam)
        │
        ▼
  pipeline_thread  ─────────────────────────────────────────┐
        │                                                     │
  ┌─────┴──────┐                                             │
  │  BGS (MOG2) │  ← always running                          │
  └─────┬──────┘                                             │
        │                                                     │
  WS client      No WS client                                │
  connected?     (offline mode)                              │
     │                │                                      │
     ▼                ▼                                      │
  YOLO on        Write motion                                │
  motion frames  clips to disk                               │
  (throttled)         │                                      │
     │           Queue clip                                  │
     │           for offline                                 │
     │           YOLO worker ◄──────── offline_yolo_worker   │
     │                                (separate thread)      │
     └─────── annotate & serve MJPEG ─────────────────────┘
                   /video_feed

WebSocket /ws   → streams detection JSON events to browser
/status         → JSON health check
/              → viewer HTML page
```

## Swapping in a Hummingbird-specific YOLO model

1. Train or download a custom model (e.g. from Roboflow hummingbird dataset)
2. Place the `.pt` file in the same directory as app.py
3. Edit `YOLO_MODEL_PATH = "your_model.pt"` at the top of app.py
4. Set `YOLO_CLASSES = None` (detect all) or restrict to specific class IDs

## Detection log

Offline detections are appended to `detections.jsonl`:
```json
{"timestamp":"2026-06-27T22:00:00","clip":"clips/clip_20260627_220000.mp4","detections":[{"label":"bird","conf":0.82,"bbox":[120,80,340,290]}]}
```
