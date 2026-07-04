from flask import Flask, Response
from picamera2 import Picamera2
import io, time

FOCUS_LOCK_DIOPTERS = 2.5

app = Flask(__name__)
camera = Picamera2()
camera.configure(camera.create_video_configuration(main={"size": (1280, 720)}))
camera.set_controls({"AfMode": 0, "LensPosition": FOCUS_LOCK_DIOPTERS})
camera.start()

def generate():
    while True:
        stream = io.BytesIO()
        camera.capture_file(stream, format='jpeg')
        stream.seek(0)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + stream.read() + b'\r\n')
        time.sleep(0.05)  # ~20fps

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '<html><body style="background:#000;margin:0"><img src="/video_feed" style="width:100%;height:100vh;object-fit:contain"></body></html>'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
