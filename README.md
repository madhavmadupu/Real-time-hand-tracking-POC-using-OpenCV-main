# 🚨 Real-Time Hand Tracking Danger Detection System

A **real-time computer vision system** that tracks hand movements and detects proximity to virtual boundaries — built **entirely with classical computer vision techniques** using OpenCV and NumPy.
No MediaPipe, OpenPose, or cloud AI APIs required.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## 🎯 Overview

This proof-of-concept demonstrates a **real-time hand tracking and safety warning system** that triggers distance-based alerts as the user’s hand approaches a defined virtual object.

Built entirely on **OpenCV + NumPy**, it achieves **15–30 FPS on CPU-only** execution.

### 🔑 Features

- ✅ Real-time hand tracking using classical computer vision
- ✅ Three-state warning system (SAFE → WARNING → DANGER)
- ✅ No external APIs or dependencies — pure OpenCV
- ✅ Motion-based filtering to isolate active hands
- ✅ Optimized for CPU performance
- ✅ Intuitive visual feedback with on-screen overlays

---

## 🎥 How It Works

### 🧠 Detection Pipeline

```

Camera Input → Skin Detection (YCrCb) → Motion Filtering →
Morphological Operations → Contour Detection → Position Filtering →
Distance Calculation → State Classification → Visual Rendering

```

### ⚙️ State System

| State      | Distance | Visual Feedback                             | Description                         |
| ---------- | -------- | ------------------------------------------- | ----------------------------------- |
| 🟢 SAFE    | > 150px  | Green boundary                              | Hand is far from the virtual object |
| 🟠 WARNING | 80–150px | Yellow/Orange boundary + “DON’T COME CLOSE” | Hand approaching                    |
| 🔴 DANGER  | ≤ 80px   | Red boundary + Full-screen alert            | “DANGER! MOVE BACK!”                |

---

## 🛠️ Technical Implementation

### Core Computer Vision Techniques

1. **YCrCb Skin Detection**
   - More stable than RGB/HSV under varied lighting
   - Adaptive thresholds for different skin tones

2. **Motion Detection**
   - Frame differencing to isolate moving regions
   - Differentiates hands (moving) from faces (static)

3. **Position-Based Filtering**
   - Prioritizes contours in lower frame regions
   - Filters out upper-frame areas (faces)

4. **Morphological Processing**
   - Erosion → remove noise
   - Dilation → close gaps
   - Gaussian blur → smooth mask edges

5. **Distance Measurement**
   - Euclidean distance from hand centroid to boundary
   - Threshold-based state classification in real time

---

## 📋 Requirements

### Dependencies

```bash
pip install opencv-python numpy
```

or

```bash
pip install -r requirements.txt
```

### System Requirements

- Python 3.8+
- Webcam
- CPU (no GPU required)
- ≥ 4GB RAM

---

## 🚀 Setup & Usage

### Option 1: Python Script

```bash
git clone https://github.com/madhavmadupu/Real-time-hand-tracking-POC-using-OpenCV.git
cd Real-time-hand-tracking-POC-using-OpenCV
pip install -r requirements.txt
python hand_tracking.py
```

### Option 2: Jupyter Notebook

```bash
jupyter notebook Hand_Tracker.ipynb
```

Run all cells sequentially, then execute the final cell to start tracking.

### Controls

| Key         | Action          |
| ----------- | --------------- |
| `Q` / `ESC` | Quit            |
| `R`         | Reset           |
| `S`         | Save screenshot |

---

## ⚙️ Configuration

### Distance Thresholds

```python
DANGER_THRESHOLD = 80
WARNING_THRESHOLD = 150
```

### Virtual Boundary

```python
BOUNDARY = {
  'x': 400,
  'y': 150,
  'width': 200,
  'height': 300
}
```

### Camera Settings

```python
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
```

### Skin Detection

```python
lower_skin = np.array([0, 133, 77], dtype=np.uint8)
upper_skin = np.array([255, 173, 127], dtype=np.uint8)
```

---

## 📊 Performance

| Metric      | Value     |
| ----------- | --------- |
| Target FPS  | ≥ 8 FPS   |
| Typical FPS | 15–30 FPS |
| Resolution  | 640×480   |
| Processing  | CPU-only  |
| Latency     | < 50 ms   |

**Tested on:**

- Intel i5 (8th Gen) / AMD Ryzen 5
- 8GB RAM

---

## 🎨 Customization Tips

### Improve Detection Accuracy

- Ensure even lighting
- Use simple, contrasting backgrounds
- Tune YCrCb thresholds per skin tone
- Modify position filters in contour selection

### Performance Optimization

```python
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
```

---

## 🧩 Troubleshooting

**Camera not opening**

```python
cap = cv2.VideoCapture(1)
```

**Face detected instead of hand**

```python
if cy < frame_height * 0.5:
    continue
```

**Low-light conditions**

```python
frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
```

**Noise sensitivity**

```python
mask = cv2.erode(mask, kernel, iterations=3)
```

---

## 📁 Project Structure

```
Real-time-hand-tracking-POC-using-OpenCV/
│
├── hand_tracking.py
├── Hand_Tracker.ipynb
├── requirements.txt
├── README.md
├── LICENSE
│
├── demo/
│   ├── demo.png
│   └── demo.gif
│
└── screenshots/
    └── screenshot_*.jpg
```

---

## 🎓 Educational Value

This project showcases fundamental classical computer vision concepts:

- Real-time video processing
- Color space transformations
- Morphological operations
- Contour-based tracking
- Distance-based state machines
- CPU optimization techniques

---

## 🚧 Limitations

- Lighting dependent
- Single-hand tracking
- Skin-tone threshold tuning required
- Works best with simple backgrounds
- Requires hand motion

---

## 🔮 Future Enhancements

- Multi-hand tracking
- Gesture recognition
- Stereo-based depth estimation
- Optional ML-based detection
- Mobile deployment
- Performance analytics
- GUI-based threshold tuning
- Data export (CSV/JSON)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

## 📄 License

MIT License — see the LICENSE file for details.

---

## 📧 Contact

**Author:** Madhav Madupu
GitHub: [https://github.com/madhavmadupu](https://github.com/madhavmadupu)

Project Repository:
[https://github.com/madhavmadupu/Real-time-hand-tracking-POC-using-OpenCV](https://github.com/madhavmadupu/Real-time-hand-tracking-POC-using-OpenCV)

---

⭐ If this project helps you, consider giving it a star!

**Built with ❤️ using classical computer vision techniques**
_No ML models were harmed in the making of this project_ 😄
