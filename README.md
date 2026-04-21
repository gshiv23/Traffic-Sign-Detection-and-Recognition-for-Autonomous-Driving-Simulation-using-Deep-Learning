# 🚗 Traffic Sign Detection & Recognition for Autonomous Driving simulation using Deep Learning

## 📌 Overview

This project presents a **deep learning–based Traffic Sign Detection and Recognition (TSDR) system** integrated with the **CARLA autonomous driving simulator**. The system performs real-time detection, classification, and decision-making for traffic signs, enabling autonomous vehicle response.

It combines:

* **YOLOv8** → Real-time traffic sign detection
* **CNN (TensorFlow/Keras)** → Traffic sign classification
* **CARLA Simulator** → Autonomous driving environment
* **Rule-Based Logic + PID Controller** → Vehicle control

---

## 🎯 Key Features

* ✅ Real-time traffic sign detection using YOLOv8
* ✅ Accurate classification using CNN
* ✅ Autonomous decision-making (STOP, SLOW, DRIVE)
* ✅ Integration with CARLA simulator
* ✅ Dynamic weather conditions (Rain, Clear, Cloudy)
* ✅ Collision handling & vehicle respawn
* ✅ Performance metrics (FPS, Accuracy, Speed)
* ✅ Automatic screenshot capture for results

---

## 🧠 System Architecture

```
Camera Input → Preprocessing → YOLO Detection → ROI Extraction → CNN Classification → Decision Module → PID Control → CARLA Simulation
```

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Libraries & Frameworks

* OpenCV
* NumPy
* TensorFlow / Keras
* Ultralytics YOLOv8
* Scikit-learn

### Tools

* CARLA Simulator
* CSV Logging
* MD5 Hashing (duplicate removal)

---

## 📂 Project Structure

```
traffic-sign-system/
│
├── data_preprocessing/
│   └── Data_Preprocessing_ITS.py
│
├── model_training/
│   └── model_training.py
│
├── carla_scripts/
│   └── run_pipeline.py
│
├── models/
│   ├── traffic_sign_final_model.h5
│   └── yolov8n.pt
│
├── screenshots/
│
└── README.md
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/traffic-sign-system.git
cd traffic-sign-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install CARLA

Download from: https://carla.org

---

## ▶️ How to Run

### Step 1: Start CARLA Simulator

```bash
./CarlaUE4.exe
```

### Step 2: Run Pipeline

```bash
python carla_scripts/run_pipeline.py
```

---

## 📊 Performance

| Metric               | Value          |
| -------------------- | -------------- |
| Accuracy             | ~85% (Offline) |
| FPS                  | 3–9 FPS        |
| Detection Confidence | > 0.8          |
| Response Time        | Real-time      |

---

## 📸 Output

The system displays:

* Bounding boxes for detected signs
* Sign label + confidence
* Vehicle decision (STOP / SLOW / DRIVE)
* FPS, Speed, Weather

Screenshots are automatically saved in the `screenshots/` folder.

---

## 🚦 Decision Logic

* **STOP Sign** → Vehicle stops
* **Speed/Warning Signs** → Reduce speed
* **No Sign** → Continue driving

---

## ⚠️ Limitations

* Performance drops in extreme weather conditions
* Accuracy depends on dataset quality
* FPS varies with hardware

---

## 🔮 Future Work

* Use advanced models (YOLOv8-large, EfficientDet)
* Deploy on real vehicles / edge devices
* Reinforcement learning for decision-making
* Multi-sensor fusion (LiDAR, Radar)

---

## 📚 References

* YOLOv8 Documentation: https://docs.ultralytics.com
* CARLA Simulator: https://carla.org
* TensorFlow: https://www.tensorflow.org

---

## 👨‍💻 Author

**Shiv Gandhi**

---

## ⭐ If you like this project, give it a star!
