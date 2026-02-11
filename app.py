import sys
import shutil
import subprocess
from pathlib import Path

from wasteDetection.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

# -------- PATHS --------
BASE_DIR = Path(__file__).resolve().parent
YOLO_DIR = BASE_DIR / "yolov5"
DATA_DIR = BASE_DIR / "data"
RUNS_DIR = YOLO_DIR / "runs"

DATA_DIR.mkdir(exist_ok=True)

# -------- CLIENT STATE --------
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"

clApp = ClientApp()

# -------- HELPERS --------
def safe_delete_runs():
    """Delete yolov5/runs safely on Windows."""
    try:
        if RUNS_DIR.exists():
            shutil.rmtree(RUNS_DIR)
    except Exception as e:
        print(f"[WARN] Could not delete runs folder: {e}")

def run_yolo_detect(source: str):
    """Run YOLOv5 detect.py with fixed output folder runs/detect/result."""
    weights_path = YOLO_DIR / "my_model.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Model not found: {weights_path}")

    detect_py = YOLO_DIR / "detect.py"
    if not detect_py.exists():
        raise FileNotFoundError(f"detect.py not found: {detect_py}")

    cmd = [
        sys.executable, "detect.py",
        "--weights", str(weights_path),
        "--img", "320",              # faster than 416
        "--conf", "0.5",
        "--source", source,
        "--project", "runs/detect",
        "--name", "result",
        "--exist-ok",
        "--device", "cpu"            # explicit CPU
    ]

    p = subprocess.run(
        cmd,
        cwd=str(YOLO_DIR),
        capture_output=True,
        text=True
    )

    if p.returncode != 0:
        raise RuntimeError(
            f"YOLO detect failed.\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )

# -------- ROUTES --------
@app.route("/train")
def trainRoute():
    # Training disabled (you already have my_model.pt)
    return "Training disabled. Use /predict with my_model.pt", 200

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@cross_origin()
def predictRoute():
    try:
        data = request.get_json(silent=True) or {}
        if "image" not in data:
            return jsonify({"error": "Send JSON with key 'image' (base64)."}), 400

        image_b64 = data["image"]

        # IMPORTANT FIX:
        # decodeImage expects only a filename (it saves into ./data internally)
        decodeImage(image_b64, clApp.filename)

        input_path = DATA_DIR / clApp.filename
        if not input_path.exists():
            return jsonify({"error": f"Failed to save image at {input_path}"}), 500

        # Clean old YOLO outputs
        safe_delete_runs()

        # Run detection
        run_yolo_detect(source=str(input_path))

        # Fixed output path
        output_path = YOLO_DIR / "runs" / "detect" / "result" / input_path.name
        if not output_path.exists():
            return jsonify({
                "error": "Detection ran but output image not found.",
                "expected": str(output_path),
                "hint": "Check yolov5/runs/detect/result/"
            }), 500

        opencodedbase64 = encodeImageIntoBase64(str(output_path))
        return jsonify({"image": opencodedbase64.decode("utf-8")})

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

@app.route("/live", methods=["GET"])
@cross_origin()
def predictLive():
    try:
        safe_delete_runs()
        run_yolo_detect(source="0")   # webcam
        return "Camera starting!!"
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

# -------- RUN SERVER (NO AIRFLOW CONFLICT) --------
if __name__ == "__main__":
    print(">>> Starting Flask on 0.0.0.0:5000", flush=True)
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
