"""
Focus Tuner — Raspberry Pi Camera Module 3
-------------------------------------------
Live MJPEG stream + web UI slider to adjust LensPosition in real time.
A live Sharpness score (Laplacian variance) peaks when focus is correct.

Usage (Pi):
    python focus_tuner.py

Usage (dev/Mac — webcam fallback, focus slider has no effect on webcam):
    python focus_tuner.py --no-camera

Open: http://hummingcam.local:5001
"""

import argparse
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except Exception:
    PICAMERA2_AVAILABLE = False

STREAM_W, STREAM_H = 1280, 720
JPEG_QUALITY       = 75
MIN_LENS_POS       = 0.0
MAX_LENS_POS       = 15.0

app = Flask(__name__)

latest_jpeg = None
jpeg_lock   = threading.Lock()
current_pos = 0.0
pos_lock    = threading.Lock()
camera_obj  = None


# ── Camera init ───────────────────────────────────────────────────────────────
def init_camera(use_pi: bool):
    global camera_obj
    if use_pi and PICAMERA2_AVAILABLE:
        cam = Picamera2()
        cfg = cam.create_video_configuration(
            main={"size": (STREAM_W, STREAM_H), "format": "RGB888"}
        )
        cam.configure(cfg)
        cam.set_controls({"AfMode": 0, "LensPosition": 0.0})
        cam.start()
        time.sleep(1.0)
        camera_obj = ("pi", cam)
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_H)
        camera_obj = ("cv2", cap)


def apply_focus(pos: float, af_mode: str = "manual"):
    global current_pos
    pos = float(np.clip(pos, MIN_LENS_POS, MAX_LENS_POS))
    with pos_lock:
        current_pos = pos
    if camera_obj and camera_obj[0] == "pi":
        cam = camera_obj[1]
        if af_mode == "manual":
            cam.set_controls({"AfMode": 0, "LensPosition": pos})
        elif af_mode == "single":
            cam.set_controls({"AfMode": 1, "AfTrigger": 0})
        elif af_mode == "continuous":
            cam.set_controls({"AfMode": 2})


def capture_rgb():
    if camera_obj is None:
        return None
    kind, cam = camera_obj
    if kind == "pi":
        return cam.capture_array()
    ret, bgr = cam.read()
    if ret:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return None


# ── Capture / encode loop ─────────────────────────────────────────────────────
def capture_loop():
    global latest_jpeg
    while True:
        rgb = capture_rgb()
        if rgb is None:
            time.sleep(0.02)
            continue

        with pos_lock:
            pos = current_pos

        sharp    = float(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
        dist_ft  = (1.0 / pos) * 3.281 if pos > 0.001 else None
        dist_str = f"{dist_ft:.1f} ft" if dist_ft else "infinity"

        h, w = rgb.shape[:2]
        overlay = rgb.copy()
        cv2.rectangle(overlay, (0, h - 52), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.72, rgb, 0.28, 0)
        cv2.putText(
            frame,
            f"LensPosition: {pos:.3f}   Distance: ~{dist_str}   Sharpness: {sharp:.0f}",
            (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2
        )

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        with jpeg_lock:
            latest_jpeg = buf.tobytes()
        time.sleep(1 / 30)


# ── HTML page ─────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Focus Tuner</title>
<style>
  :root{--bg:#0e1117;--surface:#161b22;--border:#30363d;--text:#e6edf3;
        --muted:#8b949e;--accent:#3fb950;--track:#21262d;}
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
  body{background:var(--bg);color:var(--text);font-family:system-ui,sans-serif;font-size:14px;
       min-height:100dvh;display:flex;flex-direction:column;align-items:center;padding:20px 16px;gap:20px;}
  h1{font-size:18px;font-weight:600;}
  .stream-wrap{width:100%;max-width:960px;aspect-ratio:16/9;background:#000;border-radius:10px;
               overflow:hidden;border:1px solid var(--border);box-shadow:0 8px 32px rgba(0,0,0,.5);}
  .stream-wrap img{width:100%;height:100%;object-fit:contain;display:block;}
  .controls{width:100%;max-width:960px;background:var(--surface);border:1px solid var(--border);
            border-radius:10px;padding:20px 24px;display:flex;flex-direction:column;gap:18px;}
  .metrics{display:flex;gap:28px;flex-wrap:wrap;}
  .metric .mlabel{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;}
  .metric .mval{font-size:22px;font-weight:700;font-family:monospace;color:var(--accent);margin-top:2px;}
  .row{display:flex;align-items:center;gap:14px;}
  .row label{white-space:nowrap;min-width:130px;color:var(--muted);font-size:13px;}
  input[type=range]{flex:1;height:6px;appearance:none;background:var(--track);
                    border-radius:999px;outline:none;cursor:pointer;}
  input[type=range]::-webkit-slider-thumb{appearance:none;width:18px;height:18px;border-radius:50%;
    background:var(--accent);border:2px solid var(--bg);cursor:pointer;transition:transform .15s;}
  input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.2);}
  .badge{min-width:64px;text-align:right;font-family:monospace;font-size:13px;}
  .btn-row{display:flex;gap:8px;flex-wrap:wrap;}
  button{padding:7px 14px;border-radius:6px;border:1px solid var(--border);
         background:var(--surface);color:var(--text);font-size:13px;cursor:pointer;
         transition:background .15s,border-color .15s;}
  button:hover{background:var(--track);border-color:var(--accent);}
  button.active{background:var(--accent);color:#000;border-color:var(--accent);font-weight:600;}
  hr{border:none;border-top:1px solid var(--border);}
  .copy-wrap{display:none;}
  .copy-wrap.show{display:block;}
  .copy-box{background:var(--bg);border:1px solid var(--border);border-radius:6px;
            padding:10px 14px;font-family:monospace;font-size:13px;color:var(--accent);
            margin-top:8px;word-break:break-all;}
  .hint{font-size:12px;color:var(--muted);line-height:1.5;}
  .trend{font-size:12px;margin-left:4px;}
</style>
</head>
<body>
<h1>📷 RPi Camera 3 — Focus Tuner</h1>
<div class="stream-wrap"><img src="/video_feed" alt="Live feed"></div>
<div class="controls">

  <div class="metrics">
    <div class="metric">
      <div class="mlabel">Lens Position</div>
      <div class="mval" id="m-pos">0.000</div>
    </div>
    <div class="metric">
      <div class="mlabel">Approx. Distance</div>
      <div class="mval" id="m-dist">∞</div>
    </div>
    <div class="metric">
      <div class="mlabel">Sharpness <span class="trend" id="trend"></span></div>
      <div class="mval" id="m-sharp">—</div>
    </div>
  </div>

  <hr>

  <div class="row">
    <label>Focus (LensPosition)</label>
    <input type="range" id="focus-slider" min="0" max="15" step="0.01" value="0">
    <span class="badge" id="focus-val">0.000</span>
  </div>
  <div class="row">
    <label>Fine-tune step</label>
    <input type="range" id="step-slider" min="0.001" max="0.5" step="0.001" value="0.01">
    <span class="badge" id="step-val">0.010</span>
  </div>

  <div class="btn-row">
    <button id="btn-far">◀ Farther</button>
    <button id="btn-near">Closer ▶</button>
    <button id="btn-inf">∞ Infinity</button>
    <button id="btn-10ft">~10 ft</button>
    <button id="btn-5ft">~5 ft</button>
    <button id="btn-4ft">~4 ft</button>
    <button id="btn-2ft">~2 ft</button>
    <button id="btn-1ft">~1 ft</button>
  </div>

  <hr>

  <div class="btn-row">
    <span style="color:var(--muted);font-size:13px;align-self:center;">AF Mode:</span>
    <button class="active" id="btn-manual">Manual (locked)</button>
    <button id="btn-single">Single cycle</button>
    <button id="btn-cont">Continuous</button>
  </div>
  <p class="hint"><b>Single cycle:</b> camera runs one AF pass then locks. Slider updates after ~2.5s to show where it settled — good starting point for manual fine-tuning.</p>

  <hr>

  <div class="btn-row">
    <button class="active" id="btn-lock">🔒 Lock &amp; Copy to app.py</button>
  </div>
  <div class="copy-wrap" id="copy-wrap">
    <div class="copy-box" id="copy-box"></div>
    <p class="hint" style="margin-top:6px;">Paste into <code>FOCUS_LOCK_DIOPTERS</code> in app.py. Also copied to clipboard.</p>
  </div>

</div>
<script>
const focusSlider=document.getElementById('focus-slider'),focusVal=document.getElementById('focus-val');
const stepSlider=document.getElementById('step-slider'),stepVal=document.getElementById('step-val');
const mPos=document.getElementById('m-pos'),mDist=document.getElementById('m-dist');
const mSharp=document.getElementById('m-sharp'),trend=document.getElementById('trend');
const copyWrap=document.getElementById('copy-wrap'),copyBox=document.getElementById('copy-box');
let lastSharp=null;

function distStr(pos){
  if(pos<=0.001)return'\u221e';
  const ft=(1/pos)*3.281;
  return ft>200?'\u221e':ft.toFixed(1)+' ft';
}
function updateDisplay(pos){
  pos=parseFloat(pos);
  focusSlider.value=pos;focusVal.textContent=pos.toFixed(3);
  mPos.textContent=pos.toFixed(3);mDist.textContent=distStr(pos);
}
async function applyFocus(pos,afMode='manual'){
  pos=Math.max(0,Math.min(15,parseFloat(pos)));
  updateDisplay(pos);
  await fetch('/set_focus',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({lens_position:pos,af_mode:afMode})});
}
focusSlider.addEventListener('input',()=>updateDisplay(focusSlider.value));
focusSlider.addEventListener('change',()=>applyFocus(focusSlider.value));
stepSlider.addEventListener('input',()=>stepVal.textContent=parseFloat(stepSlider.value).toFixed(3));
const step=dir=>applyFocus(parseFloat(focusSlider.value)+dir*parseFloat(stepSlider.value));
document.getElementById('btn-far').onclick=()=>step(-1);
document.getElementById('btn-near').onclick=()=>step(1);
document.getElementById('btn-inf').onclick=()=>applyFocus(0);
document.getElementById('btn-10ft').onclick=()=>applyFocus(0.1);
document.getElementById('btn-5ft').onclick=()=>applyFocus(0.2);
document.getElementById('btn-4ft').onclick=()=>applyFocus(0.25);
document.getElementById('btn-2ft').onclick=()=>applyFocus(0.5);
document.getElementById('btn-1ft').onclick=()=>applyFocus(1.0);
function setAFBtns(active){
  ['btn-manual','btn-single','btn-cont'].forEach(id=>{
    document.getElementById(id).className=id===active?'active':'';
  });
}
document.getElementById('btn-manual').onclick=()=>{setAFBtns('btn-manual');applyFocus(parseFloat(focusSlider.value),'manual');};
document.getElementById('btn-single').onclick=()=>{
  setAFBtns('btn-single');applyFocus(parseFloat(focusSlider.value),'single');
  setTimeout(async()=>{
    const r=await fetch('/status');const s=await r.json();
    if(s.lens_position!=null)updateDisplay(s.lens_position);
    setAFBtns('btn-manual');
  },2500);
};
document.getElementById('btn-cont').onclick=()=>{setAFBtns('btn-cont');applyFocus(parseFloat(focusSlider.value),'continuous');};
document.getElementById('btn-lock').onclick=()=>{
  const pos=parseFloat(focusSlider.value);
  applyFocus(pos,'manual');
  const snippet='FOCUS_LOCK_DIOPTERS = '+pos.toFixed(3)+'  # ~'+distStr(pos);
  copyBox.textContent=snippet;copyWrap.classList.add('show');
  navigator.clipboard?.writeText(snippet);
};
setInterval(async()=>{
  try{
    const r=await fetch('/status');const s=await r.json();
    if(s.sharpness!=null){
      mSharp.textContent=Math.round(s.sharpness).toLocaleString();
      if(lastSharp!==null){
        const d=s.sharpness-lastSharp;
        trend.textContent=d>50?'\u25b2':d<-50?'\u25bc':'\u2014';
        trend.style.color=d>50?'#3fb950':d<-50?'#f85149':'#8b949e';
      }
      lastSharp=s.sharpness;
    }
    if(s.lens_position!=null){
      mPos.textContent=parseFloat(s.lens_position).toFixed(3);
      mDist.textContent=distStr(s.lens_position);
    }
  }catch(e){}
},1000);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            with jpeg_lock:
                frame = latest_jpeg
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(1/30)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/set_focus", methods=["POST"])
def set_focus_route():
    data = request.get_json(force=True)
    pos  = float(np.clip(data.get("lens_position", current_pos), MIN_LENS_POS, MAX_LENS_POS))
    apply_focus(pos, data.get("af_mode", "manual"))
    return jsonify({"ok": True, "lens_position": round(pos, 4)})

@app.route("/status")
def status_route():
    with pos_lock:
        pos = current_pos
    actual = pos
    if camera_obj and camera_obj[0] == "pi":
        try:
            actual = camera_obj[1].capture_metadata().get("LensPosition", pos)
        except Exception:
            pass
    sharp = None
    with jpeg_lock:
        frame = latest_jpeg
    if frame:
        buf = np.frombuffer(frame, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            sharp = float(cv2.Laplacian(img, cv2.CV_64F).var())
    return jsonify({"lens_position": round(actual, 4),
                    "sharpness": round(sharp, 1) if sharp else None,
                    "picamera2": PICAMERA2_AVAILABLE})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=5001, type=int)
    args = parser.parse_args()
    init_camera(use_pi=not args.no_camera)
    threading.Thread(target=capture_loop, daemon=True).start()
    print(f"\nFocus Tuner -> http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, threaded=True)
