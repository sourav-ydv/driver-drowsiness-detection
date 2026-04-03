# Driver Drowsiness Detection System

A real-time driver monitoring system that detects drowsiness using computer vision and deep learning techniques.

---

## Overview

This project monitors a driver's face through a webcam and detects fatigue by analyzing multiple signals such as eye closure, yawning, head movement, and temporal behavior.

It combines traditional computer vision (EAR, MAR) with deep learning models and PERCLOS to build a robust and practical system.

---

## Key Features

* Real-time face and landmark detection using MediaPipe
* Eye Aspect Ratio (EAR) for eye closure detection
* Mouth Aspect Ratio (MAR) for yawn detection
* Head pose estimation (pitch and yaw)
* Deep learning models for eye and face classification
* PERCLOS (percentage of eye closure over time)
* Multi-level alert system (mild, warning, danger)
* Audio feedback using pygame

---

## Tech Stack

* Python
* OpenCV
* MediaPipe
* TensorFlow / Keras
* NumPy
* Pygame

---

## Project Structure

```
src/
├── day01_webcam.py
├── day02_facemesh.py
├── day03_eye_landmarks.py
├── day04_ear.py
├── day05_drowsiness_alert.py
├── day06_graduated_alerts.py
├── day13_realtime_cnn.py
└── main_system.py

models/
└── face_landmarker.task
```

---

## Setup Instructions

### 1. Clone the repository

```
git clone https://github.com/your-username/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

## Model Files

Trained models are not included due to size limitations.

Download them from:
https://drive.google.com/drive/folders/1NNw0fO7tdT_8l3B_vm7nA0eVy5eCdxni?usp=drive_link

After downloading, place them in:

```
models/
├── eye_model_finetuned.h5
├── face_model_finetuned.h5
├── face_landmarker.task
```

---

## How to Run

```
python src/main_system.py
```

Press **Q** to exit the application.

---

## System Workflow

1. Capture real-time video from webcam
2. Detect facial landmarks using MediaPipe
3. Compute EAR for eye closure
4. Compute MAR for yawning detection
5. Estimate head pose (pitch & yaw)
6. Use CNN models for additional validation
7. Track PERCLOS over time
8. Combine all signals using a fusion mechanism
9. Trigger alerts based on drowsiness level

---

## Future Scope

* Improve model generalization using custom dataset
* Deploy on embedded systems (Raspberry Pi)
* Optimize latency for real-time automotive use
* Add mobile or dashboard integration

---

## Author

Sourav Yadav
