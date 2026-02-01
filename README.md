# Sesame Seeds Detection System

A comprehensive deep learning pipeline for detecting and classifying sesame seeds using YOLOv8. This system is designed to handle automated data preparation, model training (optimized for low-spec PCs), and multi-platform inference.

## 🌟 Key Features

- **Automated Data Augmentation**: Expands small datasets using advanced image transformation techniques.
- **Auto-Annotation**: Uses OpenCV contour detection to automatically generate YOLO label files from raw images.
- **Optimized Training**: Specifically configured to run efficiently on personal computers with limited resources (CPU/low RAM).
- **Multi-Functional Inference**: Supports detection from Images, Folders, Videos, and Real-time Webcam.
- **Detailed Reporting**: Generates CSV reports and visual summaries of detection results.

---

## 📁 Project Structure

```text
├── src/                     # Source code
│   ├── run_all.py           # Master script to run the entire pipeline
│   ├── augment_data.py      # Script for data augmentation
│   ├── prepare_yolo_data.py # Auto-annotation and YOLO dataset preparation
│   ├── train_yolo.py        # Optimized training script
│   └── detect.py            # Multi-functional detection script
├── data/                    # Original raw images (Categorized by class)
├── yolov8/                  # Prepared YOLO format dataset (Generated)
├── models/                  # Saved model weights (.pt files)
├── runs/                    # Training logs and results
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### 1. Installation

It is recommended to use a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Pipeline

You can run the entire process (Preparation -> Training -> Testing) with one command:

```bash
python run_all.py
```

Or run specific steps:

```bash
python run_all.py --prepare  # Prepare data only
python run_all.py --train    # Train model only
python run_all.py --test     # Test detection only
```

---

## 🔍 Detailed Usage

### Detection (Inference)

The `src/detect.py` script is highly versatile:

**Single Image:**
```bash
python src/detect.py --image path/to/image.jpg
```

**Entire Folder:**
```bash
python src/detect.py --dir path/to/folder/ --output results/
```

**Real-time Webcam:**
```bash
python src/detect.py --webcam
```

**Model Evaluation:**
```bash
python src/detect.py --eval
```

---

## 🌐 Web Application

The system now includes a modern web dashboard for easy seed detection.

### Running the Web App

1. **Start the Backend API:**
   ```bash
   python src/api.py
   ```
   *The server will run on http://localhost:8000*

2. **Run the Frontend (Development):**
   ```bash
   cd web_app
   npm install
   npm run dev
   ```
   *The dashboard will be available at the URL provided by Vite (usually http://localhost:5173)*

### Features
- **Drag & Drop** uploads.
- **Side-by-side** comparison of original vs. detected seeds.
- **Detailed Stats**: Automated count and percentage distribution for Healthy, Black, and Rain Damaged seeds.
- **Premium UI**: Glassmorphism design with fluid animations.

---

## 🛠️ How it Works

1.  **Augmentation**: Uses `albumentations` to create variations of your seeding images, improving model robustness.
2.  **Auto-Annotation**: Instead of manual labeling, the system detects seed contours to create bounding boxes, saving hours of manual work.
3.  **Optimization**: The training script defaults to the **YOLOv8 Nano** model and uses memory-efficient settings (smaller image sizes, SGD optimizer) to ensure it runs on standard laptops.

---

## 📊 Classes
The system detects three categories of sesame seeds:
- ✅ **Healthy**
- ⚫ **Black**
- 🌧️ **Rain Damaged**

---

## 📜 Requirements
- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics (YOLOv8)
- Albumentations
- Matplotlib, Pandas, Tqdm
