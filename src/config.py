"""
Configuration Module for Head Count Project

This module centralizes all configuration parameters for the passenger counting system.
It includes settings for model paths, detection thresholds, alert configurations,
video processing parameters, and logging options.

Author: Head Count Project Team
Date: 2025-12-20
"""

import os
from pathlib import Path
from typing import Dict, Any

# Project Root Directory
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
LOGS_DIR = DATA_DIR / "logs"

# Ensure directories exist
for directory in [MODELS_DIR, INPUT_DIR, OUTPUT_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


class ModelConfig:
    """Configuration for YOLOv5 detection model"""
    
    # Model weights path (will download if not exists)
    MODEL_WEIGHTS = MODELS_DIR / "yolov5s.pt"
    
    # YOLOv5 model variant ('yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')
    MODEL_VARIANT = 'yolov5s'
    
    # Detection confidence threshold (0.0 - 1.0)
    CONFIDENCE_THRESHOLD = 0.4
    
    # IoU threshold for Non-Maximum Suppression
    IOU_THRESHOLD = 0.45
    
    # Maximum number of detections per image
    MAX_DETECTIONS = 1000
    
    # Class ID for person detection (COCO dataset)
    PERSON_CLASS_ID = 0
    
    # Device to run inference on ('cpu', 'cuda', '0', '1', etc.)
    DEVICE = 'cpu'  # Change to 'cuda' or '0' for GPU


class VideoConfig:
    """Configuration for video processing"""
    
    # Input image/video size for model (width, height)
    INPUT_SIZE = (640, 640)
    
    # Frame skip for video processing (process every Nth frame)
    # Higher values = faster processing but less accurate
    FRAME_SKIP = 1
    
    # Enable video output with annotations
    SAVE_OUTPUT_VIDEO = True
    
    # Output video codec (e.g., 'mp4v', 'XVID', 'H264')
    OUTPUT_CODEC = 'mp4v'
    
    # Output video FPS (frames per second)
    OUTPUT_FPS = 30
    
    # Draw bounding boxes on output
    DRAW_BOUNDING_BOXES = True
    
    # Bounding box color (B, G, R)
    BBOX_COLOR = (0, 255, 0)  # Green
    
    # Bounding box thickness
    BBOX_THICKNESS = 2
    
    # Text color for annotations (B, G, R)
    TEXT_COLOR = (0, 0, 255)  # Red
    
    # Text font scale
    TEXT_SCALE = 1.0
    
    # Text thickness
    TEXT_THICKNESS = 2
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization) settings
    # Helps with detection in poor lighting conditions (this was the actual cause of the head count being off)
    ENABLE_CLAHE = True
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_GRID_SIZE = (8, 8)
    
    # Gamma Correction settings
    # Applied when frame is too bright (e.g. direct sunlight)
    ENABLE_GAMMA_CORRECTION = True
    BRIGHTNESS_THRESHOLD = 0.75  # 0.0 to 1.0 (0.75 = very bright)
    GAMMA_VALUE = 0.7            # < 1.0 darkens, > 1.0 brightens


class AlertConfig:
    """Configuration for alert and notification system"""
    
    # Occupancy thresholds for different alert levels
    NORMAL_THRESHOLD = 30      # Below this: Normal operation
    WARNING_THRESHOLD = 50     # Above this: Warning level
    CRITICAL_THRESHOLD = 70    # Above this: Critical level - immediate action needed
    
    # Alert cooldown period (seconds) - prevent alert spam
    ALERT_COOLDOWN = 60
    
    # Enable console alerts
    ENABLE_CONSOLE_ALERTS = True
    
    # Enable file logging of alerts
    ENABLE_FILE_ALERTS = True
    
    # Alert log file path
    ALERT_LOG_FILE = LOGS_DIR / "alerts.log"
    
    # Enable webhook notifications (for future integration)
    ENABLE_WEBHOOK = False
    
    # Webhook URL (to be configured for production)
    WEBHOOK_URL = "http://localhost:8000/api/alerts"
    
    # Alert message templates
    ALERT_MESSAGES = {
        'normal': "Occupancy level normal: {count} passengers",
        'warning': "⚠️ WARNING: High occupancy detected - {count} passengers. Consider deploying additional vehicles.",
        'critical': "🚨 CRITICAL: Overcrowding detected - {count} passengers! Immediate action required."
    }


class AnalyticsConfig:
    """Configuration for analytics and reporting"""
    
    # Enable analytics logging
    ENABLE_ANALYTICS = True
    
    # Analytics log file
    ANALYTICS_LOG_FILE = LOGS_DIR / "analytics.csv"
    
    # Time window for statistics (seconds)
    STATS_WINDOW = 300  # 5 minutes
    
    # Enable real-time statistics display
    SHOW_REALTIME_STATS = True
    
    # Export format for reports ('csv', 'json')
    EXPORT_FORMAT = 'csv'


class LoggingConfig:
    """Configuration for application logging"""
    
    # Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    LOG_LEVEL = 'INFO'
    
    # Log file path
    LOG_FILE = LOGS_DIR / "application.log"
    
    # Log format
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Date format
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    
    # Enable console logging
    CONSOLE_LOGGING = True
    
    # Enable file logging
    FILE_LOGGING = True
    
    # Maximum log file size (bytes) before rotation
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
    
    # Number of backup log files to keep
    BACKUP_COUNT = 5


class TrainingConfig:
    """Configuration for model training (if custom training is needed)"""
    
    # Training dataset path
    DATASET_PATH = DATA_DIR / "training_dataset"
    
    # Train/validation split ratio
    TRAIN_VAL_SPLIT = 0.8
    
    # Training epochs
    EPOCHS = 50
    
    # Batch size
    BATCH_SIZE = 16
    
    # Image size for training
    IMG_SIZE = 640
    
    # Learning rate
    LEARNING_RATE = 0.01
    
    # Enable caching for faster training
    CACHE_IMAGES = True
    
    # Number of workers for data loading
    WORKERS = 4


def get_config_summary() -> Dict[str, Any]:
    """
    Get a summary of all configuration settings.
    
    Returns:
        Dict containing all configuration parameters
    """
    return {
        'model': {
            'weights': str(ModelConfig.MODEL_WEIGHTS),
            'variant': ModelConfig.MODEL_VARIANT,
            'confidence': ModelConfig.CONFIDENCE_THRESHOLD,
            'device': ModelConfig.DEVICE
        },
        'video': {
            'input_size': VideoConfig.INPUT_SIZE,
            'frame_skip': VideoConfig.FRAME_SKIP,
            'save_output': VideoConfig.SAVE_OUTPUT_VIDEO
        },
        'alerts': {
            'normal_threshold': AlertConfig.NORMAL_THRESHOLD,
            'warning_threshold': AlertConfig.WARNING_THRESHOLD,
            'critical_threshold': AlertConfig.CRITICAL_THRESHOLD
        },
        'paths': {
            'project_root': str(PROJECT_ROOT),
            'models_dir': str(MODELS_DIR),
            'output_dir': str(OUTPUT_DIR),
            'logs_dir': str(LOGS_DIR)
        }
    }


if __name__ == "__main__":
    # Print configuration summary when run directly
    import json
    print("Head Count Project - Configuration Summary")
    print("=" * 50)
    print(json.dumps(get_config_summary(), indent=2))
