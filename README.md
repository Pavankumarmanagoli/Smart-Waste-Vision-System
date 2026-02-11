# Smart Waste Vision System ♻️

An end-to-end **Computer Vision + MLOps** project for automated waste sorting using **YOLOv5**.

This project implements a **Discarded Material Identification System** that detects waste objects in images.
---

## Project Vision

Waste sorting is a practical sustainability challenge. This project demonstrates how to build a real-world AI solution that can identify discarded materials from images with high precision and support automation workflows.

---

## Why This Project Matters

- **Cutting-edge model family**: YOLOv5 is widely used for real-time object detection.
- **Social impact use-case**: applies AI to sustainability and waste management.
- **Portfolio strength**: shows full lifecycle ownership (training + deployment), not only notebook experiments.

---

## Key Capabilities

- ✅ **Object Detection Inference API** (`POST /predict`) using Base64 images.
- ✅ **Web UI** (`GET /`) for upload + prediction visualization.
- ✅ **Live Camera Trigger** (`GET /live`) to run YOLOv5 on source `0`.
- ✅ **Modular Training Pipeline**:
  - `DataIngestion`
  - `DataValidation`
  - `ModelTrainer`
- ✅ **Container-ready setup** via Docker.

---

## System Architecture

The repository includes architecture and flowchart assets in `images/`:

- `images/Architecture.png`
- `images/flowchart_1.png`
- `images/flowchart_2.png`
- `images/flowchart_3.png`

These diagrams represent the high-level lifecycle from training data to deployed inference service.

---

## Tech Stack

1. **Deep Learning**: PyTorch, YOLOv5  
2. **Computer Vision**: OpenCV  
3. **Web Framework**: Flask, Flask-CORS  
4. **Data/Scientific**: NumPy, Pandas, SciPy, Matplotlib  
5. **Containerization**: Docker  
6. **CI/CD**: GitHub Actions (target production workflow)  
7. **Cloud**: AWS (EC2, ECR, IAM, S3)  
8. **Utilities**: Dill, setuptools, gdown, PyYAML  

---

## Repository Structure

```text
Smart-Waste-Vision-System/
├── app.py
├── main.py
├── requirements.txt
├── Dockerfile
├── templates/
│   └── index.html
├── images/
│   ├── Architecture.png
│   ├── flowchart_1.png
│   ├── flowchart_2.png
│   └── flowchart_3.png
└── wasteDetection/
    ├── components/
    │   ├── data_ingestion.py
    │   ├── data_validation.py
    │   └── model_trainer.py
    ├── pipeline/
    │   └── training_pipeline.py
    ├── entity/
    ├── constant/
    ├── logger/
    ├── utils/
    └── exception/
```

---

## How the Pipeline Works

### 1) Training Pipeline (`wasteDetection/pipeline/training_pipeline.py`)

The pipeline orchestrates:

1. **Data Ingestion**
   - Downloads dataset zip from Google Drive.
   - Extracts into feature store path.

2. **Data Validation**
   - Verifies required artifacts are present (`train`, `valid`, `data.yaml`).

3. **Model Trainer**
   - Reads class count (`nc`) from `data.yaml`.
   - Creates custom YOLOv5 model config.
   - Runs YOLOv5 training.
   - Exports trained weights (`best.pt`) to expected locations.

### 2) Inference Flow (`app.py`)

1. Client sends Base64 image to `/predict`.
2. Server decodes and stores image under `./data/`.
3. Server executes `yolov5/detect.py` with configured weights.
4. Detection output is read from YOLOv5 run directory.
5. Output image is encoded back to Base64 and returned as JSON.

---

## Getting Started

### Prerequisites

- Python **3.9+** recommended.
- Git.
- A local clone of **YOLOv5** inside this project (`./yolov5`).
- Model weights for inference placed at `./yolov5/my_model.pt`.

> Note: `yolov5/` is intentionally ignored by `.gitignore`, so each developer sets it up locally.

### Installation

```bash
git clone <your-repository-url>
cd Smart-Waste-Vision-System

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate        # Windows (PowerShell)

pip install --upgrade pip
pip install -r requirements.txt
```

### Setup YOLOv5

```bash
git clone https://github.com/ultralytics/yolov5.git
# place your trained inference model at:
# yolov5/my_model.pt
```

---

## Running the App

```bash
python app.py
```

Server starts on port `5000`.

- Local URL: `http://127.0.0.1:5000/`

---

## API Reference

### `GET /`
Returns the web interface for uploading images and visualizing predictions.

### `GET /train`
Returns a status message (training route currently disabled in Flask app).

### `POST /predict`
Runs object detection on a Base64 input image.

**Request Body**

```json
{
  "image": "<base64-image-without-data-url-prefix>"
}
```

**Success Response**

```json
{
  "image": "<base64-prediction-image>"
}
```

### `GET /live`
Starts YOLOv5 webcam inference (`--source 0`) and returns a basic status string.

---

## Train the Model Programmatically

```python
from wasteDetection.pipeline.training_pipeline import TrainPipeline

pipeline = TrainPipeline()
pipeline.run_pipeline()
```

Expected dataset zip contents:

- `train/`
- `valid/`
- `data.yaml`

---

## Docker Usage

```bash
docker build -t smart-waste-vision .
docker run --rm -p 5000:5000 smart-waste-vision
```

Open: `http://127.0.0.1:5000/`

---

## CI/CD and AWS Deployment 

Production-oriented flow for this project:

1. Push code to GitHub.
2. GitHub Actions pipeline triggers.
3. Build Docker image and push to ECR.
4. Deploy/update container on AWS EC2.
5. Serve latest model app with minimal manual intervention.

---

## Troubleshooting

- **`Model not found: yolov5/my_model.pt`**
  - Ensure the weight file exists at the exact expected path.

- **`detect.py not found`**
  - Ensure YOLOv5 is cloned as `./yolov5`.

- **Prediction output image missing**
  - Inspect Flask logs for YOLO command stdout/stderr and verify YOLO run output folders.

- **PyTorch install mismatch (CPU/CUDA)**
  - Install correct wheel from: https://pytorch.org/get-started/locally/

---

## License

Add your preferred license (e.g., MIT, Apache-2.0) in a dedicated `LICENSE` file.
