# Head Count Project

**Intelligent Passenger Counting System for Public Transport Vehicles**

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![YOLOv5](https://img.shields.io/badge/YOLOv5-Ultralytics-orange.svg)

## 📋 Overview

The **Head Count Project** is an advanced computer vision system designed to monitor passenger occupancy in public transport vehicles (buses, trains, etc.) using onboard cameras. The system provides real-time passenger counting, automated alerts for overcrowding situations, and comprehensive analytics to help transportation services optimize their operations.

### Key Features

- **Real-time Passenger Detection**: Uses YOLOv5 deep learning model for accurate person detection
- **Multi-level Alert System**: Configurable thresholds (Normal, Warning, Critical) for occupancy monitoring
- **Live Camera Support**: Process live camera feeds or pre-recorded videos
- **Comprehensive Analytics**: Track statistics, peak times, and occupancy trends
- **Flexible Deployment**: CLI interface for easy integration into existing systems
- **Extensible Architecture**: Designed for future enhancements including security monitoring

## 🎯 Use Cases

### Primary Application
- **Occupancy Monitoring**: Real-time tracking of passenger counts in vehicles
- **Overcrowding Alerts**: Automatic notifications when capacity thresholds are exceeded
- **Fleet Management**: Enable dispatch centers to deploy additional vehicles during peak times
- **Emergency Response**: Quick identification of overcrowding during festivals, emergencies, or special events
- **Route Optimization**: Data-driven insights for service improvement

### Future Scope
- **Security Monitoring**: Detection of suspicious activities or unattended objects
- **Passenger Safety**: Monitoring for falls, medical emergencies, or distress situations
- **Compliance Tracking**: Ensure adherence to capacity regulations
- **Behavioral Analysis**: Study passenger flow patterns and boarding/alighting trends

## 🏗️ Architecture

```
Head_Count_Project/
├── src/
│   ├── config.py              # Configuration management
│   ├── passenger_detector.py  # YOLOv5-based detection module
│   ├── video_processor.py     # Video/camera stream processing
│   ├── alert_system.py        # Multi-level alerting system
│   ├── analytics.py           # Statistics and reporting
│   ├── main.py               # Main CLI application
│   └── train_model.py        # Custom model training
├── models/                    # Model weights storage
├── data/
│   ├── input/                # Input videos/images
│   ├── output/               # Processed outputs
│   └── logs/                 # System and analytics logs
├── tests/                    # Test files
├── docs/                     # Additional documentation
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster processing

### Setup Steps

1. **Clone or download the project**
   ```bash
   cd Head_Count_Project
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv5 model** (automatic on first run)
   The system will automatically download the YOLOv5s model on first use.

## 💻 Usage

### Basic Usage

#### Process a Video File
```bash
python src/main.py --video path/to/video.mp4
```

#### Process Live Camera Feed
```bash
python src/main.py --camera 0 --preview
```

#### Process Images
```bash
# Single image
python src/main.py --images path/to/image.jpg

# Batch processing
python src/main.py --images path/to/images_folder/
```

### Advanced Options

#### Custom Alert Thresholds
```bash
python src/main.py --video bus_video.mp4 --warning 60 --critical 80
```

#### GPU Acceleration
```bash
python src/main.py --video video.mp4 --device cuda
```

#### Frame Skipping for Faster Processing
```bash
python src/main.py --video video.mp4 --frame-skip 2  # Process every 2nd frame
```

#### Generate Analytics Report
```bash
python src/main.py --video video.mp4 --report
```

### Configuration

Edit `src/config.py` to customize:
- Detection confidence thresholds
- Alert levels and cooldown periods
- Video processing parameters
- Logging settings
- Analytics options

### Command-Line Options

```
usage: main.py [-h] [--video VIDEO | --camera CAMERA | --images IMAGES]
               [--output OUTPUT] [--no-save] [--preview]
               [--confidence CONFIDENCE] [--device DEVICE]
               [--frame-skip FRAME_SKIP] [--warning WARNING]
               [--critical CRITICAL] [--report] [--config] [--version]

Options:
  --video VIDEO         Path to video file
  --camera CAMERA       Camera index (0 for default camera)
  --images IMAGES       Path to image file or directory
  --output OUTPUT       Output path for processed video/images
  --no-save            Disable saving output video/images
  --preview            Show live preview window
  --confidence FLOAT   Detection confidence threshold (default: 0.4)
  --device DEVICE      Device: cpu, cuda, 0, 1, etc. (default: cpu)
  --frame-skip N       Process every Nth frame (default: 1)
  --warning N          Warning threshold (default: 50)
  --critical N         Critical threshold (default: 70)
  --report             Generate analytics report
  --config             Show current configuration
```

## 📊 Output

### Processed Videos/Images
- Annotated with bounding boxes around detected passengers
- Passenger count overlay
- Real-time FPS counter
- Saved to `data/output/` directory

### Alerts
- Console output with color-coded severity levels
- Log file: `data/logs/alerts.log`
- Webhook support (configurable for production)

### Analytics
- Statistics: min, max, mean, median passenger counts
- Peak occupancy times
- Trend analysis
- CSV export: `data/logs/analytics.csv`
- Text reports: `data/output/analytics_report.txt`

## 🎓 Training Custom Models

If you have your own dataset for passenger detection:

```bash
python src/train_model.py --data path/to/dataset --epochs 50 --batch 16
```

**Dataset Structure:**
```
dataset/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── labels/
    ├── img1.txt  # YOLO format annotations
    ├── img2.txt
    └── ...
```

## 🔧 Technical Details

### Detection Model
- **Architecture**: YOLOv5 (You Only Look Once v5)
- **Framework**: PyTorch
- **Default Model**: YOLOv5s (small, fast, accurate)
- **Class**: Person detection (COCO class 0)

### Performance
- **Speed**: ~30-60 FPS on GPU, ~5-15 FPS on CPU (depends on hardware)
- **Accuracy**: High precision with configurable confidence thresholds
- **Temporal Smoothing**: Reduces count fluctuations in video streams

### Alert System
- **Levels**: Normal, Warning, Critical
- **Cooldown**: Prevents alert spam
- **Channels**: Console, file logging, webhook-ready

## 📈 Future Enhancements

### Planned Features
1. **Security Extensions**
   - Suspicious activity detection
   - Unattended object detection
   - Crowd behavior analysis

2. **Advanced Analytics**
   - Passenger flow visualization
   - Heatmap generation
   - Predictive occupancy modeling

3. **Integration Capabilities**
   - REST API for remote monitoring
   - Dashboard web interface
   - Mobile app notifications
   - Integration with existing fleet management systems

4. **Performance Improvements**
   - Multi-camera support
   - Distributed processing
   - Edge device deployment (Raspberry Pi, Jetson Nano)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional detection models (YOLO v8, Faster R-CNN)
- Web-based dashboard
- Mobile application
- Enhanced analytics visualizations
- Multi-language support

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

**Head Count Project Team**
- Internship Project - MS Studies
- Date: April 2024

## 🙏 Acknowledgments

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) for the detection model
- OpenCV community for computer vision tools
- PyTorch team for the deep learning framework

## 📞 Support

For questions, issues, or suggestions:
- Create an issue in the project repository
- Contact the development team
- Check the documentation in the `docs/` directory

---

**Note**: This system is designed for monitoring and optimization purposes. Ensure compliance with privacy regulations and obtain necessary permissions before deploying cameras in public transport vehicles.
