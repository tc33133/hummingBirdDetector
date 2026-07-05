"""
Hummingbird Detection App
--------------------------
Architecture:
  - Picamera2 captures frames into a shared ring buffer (background thread)
  - When a WebSocket client is connected:
      * Real-time pipeline: BGS mask + optional YOLO on motion events
      * Annotated JPEG streamed via MJPEG to /video_feed
      * Detection metadata broadcast via WebSocket (/ws)
  - Offline pipeline (no client connected):
      * BGS runs on every frame; motion events saved as MP4 clips
      * Queues clips for YOLO inference after recording ends
      * Results logged to detections.jsonl

Usage:
  pip install flask flask-sock picamera2 opencv-python ultralytics
  python app.py [--no-camera]   # --no-camera uses webcam fallback (dev/Mac)

Routes:
  /              → viewer page (HTML)
  /video_feed    → MJPEG stream
  /ws            → WebSocket (detection events JSON)
  /status        → JSON status
"""

import argparse
import io
import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string
from flask_sock import Sock

# ── Optional: YOLO (graceful fallback if not installed) ──────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("ultralytics not installed – YOLO disabled. pip install ultralytics")

# ── Optional: picamera2 (falls back to cv2.VideoCapture on dev machines) ─────
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except Exception:
    PICAMERA2_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
STREAM_W, STREAM_H = 1280, 720
STREAM_FPS_CAP     = 30          # max MJPEG stream FPS
JPEG_QUALITY       = 70

CLIP_DIR           = Path("clips")
LOG_FILE           = Path("detections.jsonl")

# BGS tuning
BGS_HISTORY        = 500         # frames to build background model
BGS_THRESHOLD      = 40          # sensitivity (lower = more sensitive)
MOTION_MIN_AREA    = 800         # px² – ignore tiny noise
MOTION_COOL_DOWN   = 2.0         # seconds after motion stops before saving clip
MAX_CLIP_SECONDS   = 30          # hard cap on clip length

# YOLO
YOLO_MODEL_PATH    = "yolov8n.onnx"   # ONNX export for Pi (run: yolo export model=yolov8n.pt format=onnx)
YOLO_CONF          = 0.40
YOLO_CLASSES       = None           # None = all classes; set [14] for COCO "bird"
FOCUS_LOCK_DIOPTERS= 0.25           # ~4ft fixed focus

CLIP_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.lock = threading.Lock()

        # Latest annotated JPEG (bytes) for MJPEG stream
        self.latest_jpeg: bytes | None = None

        # Whether any WebSocket client is connected
        self.ws_clients: set = set()

        # Pipeline mode flags (set by pipeline thread)
        self.motion_detected: bool = False
        self.last_detections: list = []   # [{label, conf, bbox}]
        self.fps: float = 0.0

        # Clip writer state (offline mode)
        self.clip_writer: cv2.VideoWriter | None = None
        self.clip_path: str | None = None
        self.clip_start_time: float = 0.0
        self.last_motion_time: float = 0.0

        # WebSocket event queue (thread-safe)
        self.ws_event_queue: deque = deque(maxlen=50)

    @property
    def realtime_mode(self) -> bool:
        return len(self.ws_clients) > 0


state = AppState()

# ─────────────────────────────────────────────────────────────────────────────
# Camera
# ─────────────────────────────────────────────────────────────────────────────
class Camera:
    """Unified camera abstraction: picamera2 or cv2.VideoCapture."""

    def __init__(self, use_pi_camera: bool = True):
        self._pi = None
        self._cap = None

        if use_pi_camera and PICAMERA2_AVAILABLE:
            log.info("Initialising Picamera2 …")
            self._pi = Picamera2()
            cfg = self._pi.create_video_configuration(
                main={"size": (STREAM_W, STREAM_H), "format": "RGB888"}
            )
            self._pi.configure(cfg)
            self._pi.set_controls({
                "AfMode": 0,                    # manual focus
                "LensPosition": FOCUS_LOCK_DIOPTERS,
            })
            self._pi.start()
            time.sleep(1.0)                     # warm-up
        else:
            log.info("Picamera2 unavailable – using cv2.VideoCapture(0)")
            self._cap = cv2.VideoCapture(0)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  STREAM_W)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_H)
            self._cap.set(cv2.CAP_PROP_FPS, STREAM_FPS_CAP)

    def capture_rgb(self) -> np.ndarray | None:
        """Return an RGB numpy array (H, W, 3) or None on failure."""
        if self._pi:
            return self._pi.capture_array()          # already RGB888
        if self._cap:
            ret, bgr = self._cap.read()
            if ret:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return None

    def release(self):
        if self._pi:
            self._pi.stop()
        if self._cap:
            self._cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# YOLO wrapper
# ─────────────────────────────────────────────────────────────────────────────
class Detector:
    def __init__(self):
        self._model = None
        if YOLO_AVAILABLE:
            try:
                log.info(f"Loading YOLO model: {YOLO_MODEL_PATH}")
                self._model = YOLO(YOLO_MODEL_PATH)
                log.info("YOLO model loaded.")
            except Exception as e:
                log.warning(f"YOLO load failed: {e}")

    def infer(self, rgb: np.ndarray) -> list[dict]:
        """Run inference; return list of {label, conf, bbox:[x1,y1,x2,y2]}."""
        if self._model is None:
            return []
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        results = self._model(bgr, conf=YOLO_CONF, classes=YOLO_CLASSES,
                              verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label  = results.names[cls_id]
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({"label": label, "conf": conf,
                                "bbox": [x1, y1, x2, y2]})
        return detections


# ─────────────────────────────────────────────────────────────────────────────
# Background subtraction helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_bgs():
    return cv2.createBackgroundSubtractorMOG2(
        history=BGS_HISTORY, varThreshold=BGS_THRESHOLD, detectShadows=False
    )


def apply_bgs(bgs, rgb: np.ndarray) -> tuple[np.ndarray, list[tuple], np.ndarray]:
    """
    Returns:
        mask        – binary uint8 foreground mask
        contours    – list of contours above MOTION_MIN_AREA
        mask_rgb    – 3-channel mask suitable for overlay
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = bgs.apply(gray)

    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in cnts if cv2.contourArea(c) > MOTION_MIN_AREA]

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return mask, contours, mask_rgb


def annotate_frame(rgb: np.ndarray, contours: list,
                   detections: list[dict]) -> np.ndarray:
    """Draw motion contours (green) and YOLO boxes (yellow) on a copy."""
    out = rgb.copy()

    # Motion bounding rects in green
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # YOLO detections in yellow with label
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        label = f"{d['label']} {d['conf']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 220, 0), 2)
        cv2.putText(out, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 2)

    return out


def frame_to_jpeg(rgb: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes()


def log_detections(clip_path: str, detections: list[dict]):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "clip":      clip_path,
        "detections": detections
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    log.info(f"Logged {len(detections)} detections for {clip_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Offline clip YOLO worker  (runs in its own thread)
# ─────────────────────────────────────────────────────────────────────────────
clip_queue: deque[str] = deque()
clip_queue_lock = threading.Lock()


def offline_yolo_worker(detector: Detector):
    """Pulls clip paths from clip_queue and runs YOLO frame-by-frame."""
    while True:
        path = None
        with clip_queue_lock:
            if clip_queue:
                path = clip_queue.popleft()
        if path is None:
            time.sleep(2)
            continue

        log.info(f"[Offline YOLO] Processing {path}")
        cap = cv2.VideoCapture(path)
        all_detections = []
        frame_idx = 0

        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            # Sample every 5th frame for speed
            if frame_idx % 5 == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                dets = detector.infer(rgb)
                all_detections.extend(dets)
            frame_idx += 1

        cap.release()
        # Deduplicate by label
        unique = {d["label"]: d for d in all_detections}
        log_detections(path, list(unique.values()))


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline thread
# ─────────────────────────────────────────────────────────────────────────────
def pipeline_thread(camera: Camera, detector: Detector):
    bgs = make_bgs()
    fps_counter = 0
    fps_time    = time.time()
    frame_delay = 1.0 / STREAM_FPS_CAP

    # YOLO throttle: run at most every N seconds in realtime mode
    last_yolo_time = 0.0
    YOLO_INTERVAL  = 0.5   # seconds

    log.info("Pipeline thread started.")

    while True:
        t0 = time.time()

        rgb = camera.capture_rgb()
        if rgb is None:
            time.sleep(0.01)
            continue

        # ── Background subtraction ──────────────────────────────────────────
        mask, contours, _ = apply_bgs(bgs, rgb)
        motion = len(contours) > 0
        now    = time.time()

        with state.lock:
            state.motion_detected = motion

        # ── Realtime mode (WebSocket client connected) ──────────────────────
        if state.realtime_mode:
            detections = []

            # Run YOLO on motion frames, throttled
            if motion and (now - last_yolo_time) > YOLO_INTERVAL:
                detections = detector.infer(rgb)
                last_yolo_time = now
                if detections:
                    evt = {
                        "event":      "detection",
                        "timestamp":  datetime.utcnow().isoformat(),
                        "detections": detections,
                        "motion":     motion,
                    }
                    with state.lock:
                        state.ws_event_queue.append(json.dumps(evt))
                        state.last_detections = detections

            annotated = annotate_frame(rgb, contours, detections)
            jpeg = frame_to_jpeg(annotated)

        # ── Offline mode (no client) ────────────────────────────────────────
        else:
            jpeg = frame_to_jpeg(rgb)

            if motion:
                state.last_motion_time = now

                # Start a new clip if not already recording
                with state.lock:
                    if state.clip_writer is None:
                        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                        path = str(CLIP_DIR / f"clip_{ts}.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        bgr_for_writer = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        h, w = bgr_for_writer.shape[:2]
                        state.clip_writer  = cv2.VideoWriter(path, fourcc, 20, (w, h))
                        state.clip_path    = path
                        state.clip_start_time = now
                        log.info(f"[Offline] Clip started: {path}")

            # Write frame to clip if recording
            with state.lock:
                if state.clip_writer is not None:
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    state.clip_writer.write(bgr)
                    clip_age = now - state.clip_start_time
                    idle     = now - state.last_motion_time

                    # Stop clip if cool-down exceeded or max length reached
                    if idle > MOTION_COOL_DOWN or clip_age > MAX_CLIP_SECONDS:
                        state.clip_writer.release()
                        path = state.clip_path
                        state.clip_writer = None
                        state.clip_path   = None
                        log.info(f"[Offline] Clip saved: {path}")
                        with clip_queue_lock:
                            clip_queue.append(path)

        # ── Update shared JPEG ───────────────────────────────────────────────
        with state.lock:
            state.latest_jpeg = jpeg

        # ── FPS counter ──────────────────────────────────────────────────────
        fps_counter += 1
        if (now - fps_time) >= 2.0:
            with state.lock:
                state.fps = fps_counter / (now - fps_time)
            fps_counter = 0
            fps_time    = now

        # ── Maintain target FPS ──────────────────────────────────────────────
        elapsed = time.time() - t0
        sleep   = frame_delay - elapsed
        if sleep > 0:
            time.sleep(sleep)


# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────
app  = Flask(__name__)
sock = Sock(app)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hummingbird Detector</title>
<style>
  :root { --bg:#0e1117; --surface:#161b22; --border:#30363d;
          --text:#e6edf3; --muted:#8b949e; --accent:#3fb950;
          --warn:#d29922; --error:#f85149; }
  *, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:system-ui,sans-serif;
         font-size:14px; min-height:100dvh; display:flex; flex-direction:column; }
  header { padding:12px 20px; background:var(--surface); border-bottom:1px solid var(--border);
           display:flex; align-items:center; gap:16px; }
  header h1 { font-size:16px; font-weight:600; }
  .badge { padding:2px 8px; border-radius:999px; font-size:11px; font-weight:600;
           background:var(--border); color:var(--muted); }
  .badge.live  { background:#1a3a1a; color:var(--accent); }
  .badge.motion{ background:#3a2a00; color:var(--warn); }
  main { flex:1; display:grid; grid-template-columns:1fr 300px; gap:0; overflow:hidden; }
.stream-wrap { position:relative; background:#000; display:flex; align-items:flex-start;
                  justify-content:center; overflow:hidden; min-height:400px; }
  .stream-wrap img { width:100%; height:100%; object-fit:contain; display:block; }
  .fps-badge { position:absolute; top:10px; left:10px; background:rgba(0,0,0,.6);
               padding:3px 8px; border-radius:6px; font-size:11px; color:#fff; font-family:monospace; }
  .sidebar { background:var(--surface); border-left:1px solid var(--border);
             display:flex; flex-direction:column; overflow:hidden; }
  .sidebar h2 { padding:12px 16px; font-size:13px; color:var(--muted); font-weight:500;
                border-bottom:1px solid var(--border); text-transform:uppercase; letter-spacing:.05em; }
  #event-log { flex:1; overflow-y:auto; padding:8px; display:flex;
               flex-direction:column; gap:6px; }
  .event { background:var(--bg); border:1px solid var(--border); border-radius:6px;
           padding:8px 10px; }
  .event .time { color:var(--muted); font-size:11px; font-family:monospace; }
  .event .label { font-weight:600; color:var(--accent); margin-top:2px; }
  .event .conf  { color:var(--muted); font-size:11px; }
  .no-events { color:var(--muted); font-size:12px; padding:16px; text-align:center; }
  @media(max-width:700px){ main{ grid-template-columns:1fr; grid-template-rows:auto 200px; } }
</style>
</head>
<body>
<header>
  <h1>🐦 Hummingbird Detector</h1>
  <span id="conn-badge"  class="badge">Connecting…</span>
  <span id="motion-badge" class="badge" style="display:none">Motion</span>
</header>
<main>
  <div class="stream-wrap">
    <img id="stream" src="/video_feed" alt="Camera feed">
    <div class="fps-badge" id="fps-badge">-- fps</div>
  </div>
  <aside class="sidebar">
    <h2>Detection Events</h2>
    <div id="event-log"><p class="no-events">Waiting for detections…</p></div>
  </aside>
</main>
<script>
const connBadge   = document.getElementById('conn-badge');
const motionBadge = document.getElementById('motion-badge');
const fpsBadge    = document.getElementById('fps-badge');
const eventLog    = document.getElementById('event-log');

let noEvents = true;

function addEvent(data) {
  if (noEvents) { eventLog.innerHTML = ''; noEvents = false; }
  const el = document.createElement('div');
  el.className = 'event';
  const t = new Date(data.timestamp).toLocaleTimeString();
  const labels = (data.detections || [])
    .map(d => `<div class="label">${d.label}</div><div class="conf">conf ${(d.conf*100).toFixed(0)}%</div>`)
    .join('');
  el.innerHTML = `<div class="time">${t}</div>${labels || '<div class="label" style="color:var(--warn)">Motion only</div>'}`;
  eventLog.prepend(el);
  // Keep last 50 events
  while (eventLog.children.length > 50) eventLog.removeChild(eventLog.lastChild);
}

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);

  ws.onopen = () => {
    connBadge.textContent = 'Live';
    connBadge.className   = 'badge live';
  };

  ws.onmessage = ({data}) => {
    try {
      const msg = JSON.parse(data);
      if (msg.event === 'detection') {
        addEvent(msg);
        if (msg.motion) {
          motionBadge.style.display = '';
          clearTimeout(motionBadge._t);
          motionBadge._t = setTimeout(() => motionBadge.style.display='none', 3000);
        }
      }
      if (msg.fps !== undefined) fpsBadge.textContent = msg.fps.toFixed(1) + ' fps';
    } catch(e) {}
  };

  ws.onclose = () => {
    connBadge.textContent = 'Reconnecting…';
    connBadge.className   = 'badge';
    setTimeout(connect, 2000);
  };
}
connect();

// Poll FPS from status endpoint every 2s
setInterval(async () => {
  try {
    const r = await fetch('/status');
    const s = await r.json();
    fpsBadge.textContent = s.fps.toFixed(1) + ' fps';
  } catch(e) {}
}, 2000);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with state.lock:
                frame = state.latest_jpeg
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + frame + b"\r\n")
            time.sleep(1.0 / STREAM_FPS_CAP)

    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@sock.route("/ws")
def ws_handler(ws):
    """Registers client; pumps queued events to it."""
    log.info("WebSocket client connected – switching to realtime mode.")
    with state.lock:
        state.ws_clients.add(ws)
    try:
        while True:
            # Send any queued events
            events = []
            with state.lock:
                while state.ws_event_queue:
                    events.append(state.ws_event_queue.popleft())
            for evt in events:
                try:
                    ws.send(evt)
                except Exception:
                    break

            # Send periodic FPS heartbeat
            with state.lock:
                fps = state.fps
            try:
                ws.send(json.dumps({"event": "fps", "fps": fps}))
            except Exception:
                break

            time.sleep(0.25)
    finally:
        with state.lock:
            state.ws_clients.discard(ws)
        log.info("WebSocket client disconnected – returning to offline mode.")


@app.route("/status")
def status():
    with state.lock:
        return jsonify({
            "realtime_mode":    state.realtime_mode,
            "motion_detected":  state.motion_detected,
            "fps":              round(state.fps, 1),
            "last_detections":  state.last_detections,
            "ws_clients":       len(state.ws_clients),
            "yolo_available":   YOLO_AVAILABLE,
            "picamera2":        PICAMERA2_AVAILABLE,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-camera", action="store_true",
                        help="Use webcam instead of picamera2 (dev mode)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=5000, type=int)
    args = parser.parse_args()

    use_pi = not args.no_camera
    camera   = Camera(use_pi_camera=use_pi)
    detector = Detector()

    # Start offline YOLO worker
    t_yolo = threading.Thread(target=offline_yolo_worker, args=(detector,),
                               daemon=True, name="yolo-worker")
    t_yolo.start()

    # Start main pipeline
    t_pipe = threading.Thread(target=pipeline_thread, args=(camera, detector),
                               daemon=True, name="pipeline")
    t_pipe.start()

    log.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)
