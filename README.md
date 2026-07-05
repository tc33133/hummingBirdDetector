# Hummingbird Detector

## Setup (Raspberry Pi)

```bash
# System packages
sudo apt update
sudo apt install -y python3-picamera2 python3-libcamera python3-opencv python3-venv

# Python packages (use venv with system-site-packages for picamera2)
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install flask flask-sock ultralytics

# Run
python app.py
```

Then open http://hummingcam.local:5000 in your browser.

### NCNN acceleration (optional, ~2-3x faster YOLO)

```bash
pip install ncnn
yolo export model=yolov8n.pt format=ncnn  # creates yolov8n_ncnn_model/
```

Edit `app.py`: change `YOLO_MODEL_PATH = "yolov8n_ncnn_model"` (directory, not .pt file).

## Dev mode (Mac / no Pi camera)

```bash
pip install flask flask-sock opencv-python ultralytics
python app.py --no-camera
```

## Focus tuning (Pi Camera Module 3)

```bash
python focus_tuner.py
```
Open http://hummingcam.local:5001 — use slider to find sharpest focus, click **Lock & Copy to app.py**, paste value into `FOCUS_LOCK_DIOPTERS` in `app.py`.

## Auto-start on boot (systemd)

```bash
sudo cp hummingbird.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now hummingbird.service
```

View logs: `journalctl -u hummingbird -f`

## Architecture

```
Camera (picamera2 or webcam)
        │
        ▼
  pipeline_thread ─────────────────────────────────────────┐
        │                                                     │
  ┌─────┴──────┐                                             │
  │  BGS (MOG2) │  ← always running                          │
  └─────┬──────┘                                             │
        │                                                     │
  WS client      No WS client                                │
  connected?     (offline mode)                              │
     │                │                                      │
     ▼                ▼                                      │
  YOLO on        Write motion                               │
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
/               → viewer HTML page
```

## Swapping in a Hummingbird-specific YOLO model

1. Train or download a custom model (e.g. from Roboflow hummingbird dataset)
2. Place the `.pt` file in the same directory as `app.py`
3. Edit `YOLO_MODEL_PATH = "your_model.pt"` at the top of `app.py`
4. Set `YOLO_CLASSES = None` (detect all) or restrict to specific class IDs

## Detection log

Offline detections are appended to `detections.jsonl`:

```json
{"timestamp":"2026-06-27T22:00:00","clip":"clips/clip_20260627_220000.mp4","detections":[{"label":"bird","conf":0.82,"bbox":[120,80,340,290]}]}
```

## Key config (in `app.py`)

| Constant | Default | Notes |
|---|---|---|
| `YOLO_MODEL_PATH` | `yolov8n.pt` | Swap for custom hummingbird model |
| `YOLO_CLASSES` | `None` (all) | Set `[14]` for COCO "bird" only |
| `FOCUS_LOCK_DIOPTERS` | `0.25` | ~4ft fixed focus; use `focus_tuner.py` |
| `STREAM_W/STREAM_H` | `1280x720` | Camera resolution |
| `BGS_THRESHOLD` | `40` | Motion sensitivity (lower = more) |
| `MOTION_MIN_AREA` | `800` | px² — filters noise |