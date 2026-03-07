# AI-Based Integrated Video Analytics System

A high-performance AI video analytics dashboard designed for real-time and file-based processing. The system integrates advanced object detection, facial analysis, and regional optimizations for robust deployment in diverse traffic and crowd scenarios.


## 🚀 Features

### 1. Intelligent Vehicle Analytics
- **High-Precision Detection**: Powered by **YOLOv8-Large** for superior accuracy in detecting cars, motorcycles, buses, and trucks.
- **Indian Traffic Optimization**: Specially tuned non-maximum suppression (NMS) and confidence thresholds to handle dense traffic and overlapping vehicles.
- **ANPR (Automatic Number Plate Recognition)**: 
  - Robust license plate detection and character extraction.
  - **Indian Standard Validation**: Built-in Regex for regional plate formats (e.g., MH, DL, KA, UP).
  - **HSRP Ready**: Advanced image preprocessing (CLAHE) to handle highly reflective plates.

### 2. Facial Recognition & Analysis
- **RetinaFace Integration**: The gold standard for robust face detection in varied orientations and lighting conditions.
- **Deep Analysis**: High-resolution feature extraction using **Facenet512** to identify Age, Gender, Emotion, and Race.

### 3. Crowd Intelligence
- **People Counter**: Real-time human detection and counting.
- **Density Estimation**: Automatic classification of crowd levels (Low, Medium, High, Very High).
- **Dynamic Heatmaps**: Visual representation of spatial crowd density for hotspot identification.

### 4. Advanced Dashboard
- **Live Sync Analysis**: Side-by-side video playback and real-time AI dashboard updates.
- **Unified Video Analysis**: Process entire video files through all 4 modules simultaneously with aggregated results.
- **Hardware Acceleration**: Automatic NVIDIA GPU (CUDA) detection with fallback to multi-core CPU optimization.

## 🛠️ Tech Stack

- **Backend**: Python, Flask, Gunicorn
- **AI/ML**: YOLOv8 (Ultralytics), DeepFace, EasyOCR, RetinaFace
- **Processing**: OpenCV, NumPy, ThreadPoolExecutor (Parallel Processing)
- **Frontend**: Modern Glassmorphism UI (HTML5, CSS3, JavaScript, Chart.js)

## 📦 Installation

1. **Clone the project**
   ```bash
   git clone https://github.com/abhinai2244/AI-Based-Integrated-Video-Analytics-System.git
   cd AI-Based-Integrated-Video-Analytics-System
   ```

2. **Setup virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```
   *Note: AI model weights (~300MB+) will be downloaded automatically on the first run.*

## 🚀 Deployment

The project is optimized for **Heroku** and other cloud platforms:
- **Procfile**: Multi-worker Gunicorn setup.
- **Hardware-Aware**: Dynamically scales between CPU and GPU environments.
- **Headless Optimized**: Uses `opencv-python-headless` for server-side stability.

## 📁 Project Structure

```text
├── app.py                # Central Flask Server logic
├── modules/              # Core AI Engines (ANPR, Vehicles, Faces, People)
├── templates/            # Glassmorphic UI Pages
├── static/               # CSS, JS, and Architecture Images
├── PROJECT_OVERVIEW.txt  # Detailed technical breakdown
└── requirements.txt      # Project dependencies
```

## 📄 License
This project is for educational and professional video analytics prototyping.
