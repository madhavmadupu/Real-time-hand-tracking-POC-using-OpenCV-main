# 🚨 Real-Time Hand Tracking Danger Detection System

A real-time computer vision system that tracks hand movements and detects proximity to virtual boundaries using **classical CV techniques only** - no MediaPipe, OpenPose, or cloud AI APIs required.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)


## 🎯 Project Overview

This proof-of-concept demonstrates a hand tracking system that triggers distance-based warnings when a user's hand approaches a virtual object on screen. Built entirely with OpenCV and NumPy, it achieves **real-time performance (15-30 FPS)** on CPU-only execution.

### Key Features

- ✅ **Real-time hand tracking** using classical computer vision
- ✅ **Three-state warning system** (SAFE → WARNING → DANGER)
- ✅ **No external APIs** - pure OpenCV implementation
- ✅ **Motion-based filtering** to distinguish hands from faces
- ✅ **CPU-optimized** for accessible deployment
- ✅ **Visual feedback** with live overlays and warnings

## 🎥 How It Works

### Detection Pipeline
```
Camera Input → Skin Detection (YCrCb) → Motion Filtering → 
Morphological Operations → Contour Detection → Position Filtering → 
Distance Calculation → State Classification → Visual Rendering
```

### State System

| State | Distance | Visual Feedback | Description |
|-------|----------|----------------|-------------|
| 🟢 **SAFE** | > 150px | Green boundary | Hand is far from virtual object |
| 🟠 **WARNING** | 80-150px | Orange/Yellow boundary + warning text | Hand approaching - "DON'T COME CLOSE" |
| 🔴 **DANGER** | ≤ 80px | Red boundary + full-screen alert | "DANGER! MOVE BACK!" |

## 🛠️ Technical Implementation

### Core Techniques

1. **YCrCb Color Space Skin Detection**
   - More robust than RGB/HSV for varying lighting conditions
   - Adaptive thresholding for different skin tones

2. **Motion Detection**
   - Frame differencing to isolate moving objects
   - Helps distinguish active hands from static faces

3. **Position-Based Filtering**
   - Prioritizes lower-frame contours (hands typically positioned lower)
   - Filters out upper-center regions (where faces usually are)

4. **Morphological Operations**
   - Erosion to remove noise
   - Dilation to fill gaps
   - Gaussian blur for smooth edges

5. **Distance Calculation**
   - Euclidean distance from hand centroid to nearest boundary edge
   - Real-time state classification based on threshold values

## 📋 Requirements

### Dependencies
```bash
pip install opencv-python numpy
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

### System Requirements

- Python 3.8+
- Webcam/Camera
- CPU (no GPU required)
- 4GB RAM minimum

## 🚀 Installation & Usage

### Option 1: Python Script

1. Clone the repository:
```bash
git clone https://github.com/parthrkunkunkar-ds/Real-time-hand-tracking-POC-using-OpenCV.git
cd Real-time-hand-tracking-POC-using-OpenCV
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the script:
```bash
python hand_tracking.py
```

### Option 2: Jupyter Notebook

1. Launch Jupyter:
```bash
jupyter notebook Hand_Tracker.ipynb
```

2. Run all cells sequentially

3. Execute the final cell to start tracking

### Controls

| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit application |
| `R` | Reset system |
| `S` | Save screenshot |

## ⚙️ Configuration

Customize the system by modifying these parameters:

### Distance Thresholds
```python
# in hand_tracking.py or notebook cell

DANGER_THRESHOLD = 80    # Distance (px) for DANGER state
WARNING_THRESHOLD = 150  # Distance (px) for WARNING state
# > 150px automatically becomes SAFE
```

### Virtual Boundary
```python
BOUNDARY = {
    'x': 400,      # X position
    'y': 150,      # Y position
    'width': 200,  # Width
    'height': 300  # Height
}
```

### Camera Settings
```python
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
```

### Skin Detection Tuning
```python
# YCrCb color space ranges
lower_skin = np.array([0, 133, 77], dtype=np.uint8)
upper_skin = np.array([255, 173, 127], dtype=np.uint8)

# HSV alternative (less robust but adjustable)
lower_skin_hsv = np.array([0, 20, 70], dtype=np.uint8)
upper_skin_hsv = np.array([20, 255, 255], dtype=np.uint8)
```

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Target FPS** | ≥8 FPS |
| **Typical FPS** | 15-30 FPS |
| **Resolution** | 640x480 (default) |
| **Processing** | CPU-only |
| **Latency** | <50ms |

Tested on:
- Intel i5 (8th gen) / AMD Ryzen 5
- 8GB RAM
- Integrated graphics

## 🎨 Customization Tips

### Improve Detection Accuracy

1. **Better Lighting**: Ensure even lighting on hands
2. **Contrasting Background**: Use plain backgrounds
3. **Adjust Thresholds**: Fine-tune YCrCb ranges for your skin tone
4. **Position Filtering**: Modify position scoring in `find_largest_contour()`

### Optimize Performance
```python
# Reduce resolution for faster processing
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

# Adjust morphological kernel sizes
kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: Camera not opening
```python
# Try different camera indices
cap = cv2.VideoCapture(1)  # or 2, 3, etc.
```

**Issue**: Face detected instead of hand
```python
# Increase position filtering strictness
if cy < frame_height * 0.5:  # Ignore upper 50%
    continue
```

**Issue**: Poor detection in low light
```python
# Adjust brightness/contrast
frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
```

**Issue**: Too sensitive to noise
```python
# Increase erosion iterations
mask = cv2.erode(mask, kernel_erode, iterations=3)
```

## 📁 Project Structure
```
Real-time-hand-tracking-POC-using-OpenCV/
│
├── hand_tracking.py          # Main Python script
├── Hand_Tracker.ipynb        # Jupyter notebook version
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── LICENSE                   # MIT License
│
├── demo/                     # Demo images/videos
│   ├── demo.png
│   └── demo.gif
│
└── screenshots/              # Saved screenshots (generated)
    └── screenshot_*.jpg
```

## 🔬 Technical Details

### Algorithm Breakdown

#### 1. Skin Detection (YCrCb)
```python
ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
```
- **Why YCrCb?** More invariant to lighting changes than RGB
- **Cr, Cb channels** capture chrominance (color) independent of brightness

#### 2. Motion Filtering
```python
frame_diff = cv2.absdiff(prev_frame, gray)
motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
combined = cv2.bitwise_and(skin_mask, motion_mask)
```
- Reduces false positives from static skin-colored objects
- Focuses on actively moving hands

#### 3. Morphological Operations
```python
mask = cv2.erode(mask, kernel, iterations=2)   # Remove noise
mask = cv2.dilate(mask, kernel, iterations=2)  # Fill gaps
mask = cv2.GaussianBlur(mask, (5, 5), 0)      # Smooth
```
- Opens (erode→dilate) removes small noise
- Closes (dilate→erode) fills small holes

#### 4. Contour Detection
```python
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest = max(contours, key=cv2.contourArea)
```
- Extracts connected components
- Largest area assumed to be hand

## 🎓 Educational Value

This project demonstrates:

- Classical computer vision fundamentals
- Real-time video processing
- Color space transformations
- Morphological image processing
- Contour detection and analysis
- Distance-based state machines
- Performance optimization techniques

Perfect for:
- Computer vision students
- OpenCV beginners
- Real-time processing learners
- Developers avoiding ML dependencies

## 🚧 Limitations

- **Lighting Dependent**: Requires adequate lighting for skin detection
- **Single Hand**: Tracks only one hand at a time
- **Skin Tone Bias**: May need calibration for different skin tones
- **Background Sensitivity**: Works best with contrasting backgrounds
- **Motion Required**: Static hands may not be detected (if motion filtering enabled)

## 🔮 Future Enhancements

- [ ] Multi-hand tracking support
- [ ] Gesture recognition (thumb up, peace sign, etc.)
- [ ] 3D distance estimation using stereo vision
- [ ] Machine learning-based hand detection
- [ ] Mobile app deployment (Android/iOS)
- [ ] Real-time performance graphs
- [ ] Configurable UI for threshold tuning
- [ ] Export tracking data to CSV/JSON

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Improvement

- Better hand/face discrimination
- Improved skin detection for diverse skin tones
- Performance optimizations
- Additional gesture recognition
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV community for excellent documentation
- Computer vision researchers for classical CV techniques
- Inspiration from industrial safety systems

## 📧 Contact

Madhav Madupu - [@madhavmadupu](https://github.com/madhavmadupu)

Project Link: [https://github.com/madhavmadupu/Real-time-hand-tracking-POC-using-OpenCV](https://github.com/madhavmadupu/Real-time-hand-tracking-POC-using-OpenCV)

---

## ⭐ Star History

If you find this project useful, please consider giving it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=madhavmadupu/Real-time-hand-tracking-POC-using-OpenCV&type=Date)](https://star-history.com/#madhavmadupu/Real-time-hand-tracking-POC-using-OpenCV&Date)

---

**Built with ❤️ using classical Computer Vision techniques**

*No ML models were harmed in the making of this project* 😄
