# Recommended Production ML Models

**Project**: Crowd Safety & ANPR Testbed

---

## 📌 Executive Summary

This directory contains the **curated top-performing ML models** selected from empirical benchmarks across fall detection, ANPR/license plate OCR, umbrella crowd density, violence classification, and traffic monitoring.

---

## 🚀 Top Recommended Models by Domain

| Category | Recommended Model | Weight File / Path | Key Strength & Performance |
| :--- | :--- | :--- | :--- |
| **ANPR (OCR Engine)** | **RapidOCR (PP-OCRv4 ONNX)** | `auto-download / ONNX Runtime` | **+17% higher plate read accuracy** (325 reads vs 278 EasyOCR) & 2.5x faster CPU inference. |
| **ANPR (Vehicle Detector)** | **Indian ANPR Pipeline** | `ocr_and_anpr/detr_plate/` & `easyocr/` | **High vehicle recall** (1,886 captures), detecting auto-rickshaws, tempos, and 2-wheelers. |
| **Umbrellas (Dense Crowds)** | **RF-DETR Nano (DINOv2)** | `weights/rtdetr-l.pt` | **Top recall in dense crowds** (20,030 detections, 240 tracks). Transformer attention resolves overlapping umbrellas. |
| **Umbrellas (Edge / Real-Time)** | **YOLO26-Nano** | `weights/yolo11n.pt` | **NMS-free end-to-end** (<10 MB footprint, 3.2x faster CPU/Jetson inference). |
| **Fall Detection** | **YOLOv8-Pose + Heuristic** | `weights/yolov8s-pose.pt` | **Best posture accuracy & tracking** (5,277 posture frames, 229 verified falls, 0.854 avg conf). |
| **Violence Detection** | **Roboflow Surveillance Model** | `API Hosted / weights/slowfast_r50_kinetics400.pt` | **398 fight alerts (0.654 conf)** trained on real surveillance fight footage. |
| **Traffic Monitoring** | **RT-DETR-L Traffic** | `weights/rtdetr-l.pt` | **Best tracking continuity** (391 unique vehicle IDs, 9,926 detections on distant/occluded cars). |

---

## 📂 Directory Layout

```
Recommended_Models/
├── RECOMMENDED_MODELS_GUIDE.md   # This decision guide
├── weights/
│   ├── yolov8s-pose.pt           # Fall Detection keypoint backbone
│   ├── rtdetr-l.pt               # RT-DETR Traffic & RF-DETR Umbrella transformer
│   ├── yolo11n.pt                # Edge Traffic & Edge Umbrella detector
│   ├── yolov8s-worldv2.pt        # Open-vocabulary umbrella detector
│   └── pose_landmarker_full.task # MediaPipe BlazePose model
├── ocr_and_anpr/
│   ├── detr_plate/               # HuggingFace DETR license plate detector
│   └── easyocr/                  # CRAFT & English G2 OCR weights
└── video_classifiers/
    ├── x3d_s_kinetics400.pt      # Lightweight X3D-S violence classifier
    └── slowfast_r50_kinetics400.pt # Dual-pathway SlowFast-R50 violence classifier
```

---

## 🛠 Integration Quickstart

```python
# 1. Fall Detection
from models.fall import YOLOPoseFallDetector
fall_detector = YOLOPoseFallDetector(model_size="s", device="cuda")

# 2. ANPR & Number Plate Reading
from models.anpr import ANPRDetector
anpr_pipeline = ANPRDetector(ocr_backend="rapidocr", device="cuda")

# 3. Dense Crowd Umbrella Detection
from models.umbrella import RFDETRNanoUmbrellaDetector
umbrella_detector = RFDETRNanoUmbrellaDetector(device="cuda")

# 4. Traffic Monitoring
from models.traffic import RtdetrTrafficDetector
traffic_detector = RtdetrTrafficDetector(device="cuda")
```
