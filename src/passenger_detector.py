"""
Passenger Detection Module

This module implements the core passenger detection functionality using YOLOv5.
It provides real-time person detection and counting capabilities for monitoring
passenger occupancy in public transport vehicles.

Key Features:
- YOLOv5-based person detection
- Confidence-based filtering
- Temporal smoothing for stable counts
- Batch processing support
- Detailed detection metadata

Author: Head Count Project Team
Date: 2025-12-20
"""

import torch
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import logging
from collections import deque

from config import ModelConfig, VideoConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PassengerDetector:
    """
    Passenger detection class using YOLOv5 for person detection.
    
    This class handles loading the YOLOv5 model, running inference on images/frames,
    and extracting passenger count information with confidence scores.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        confidence_threshold: float = None,
        device: str = None,
        use_temporal_smoothing: bool = True,
        smoothing_window: int = 5
    ):
        """
        Initialize the passenger detector.
        
        Args:
            model_path: Path to YOLOv5 model weights. If None, uses config default.
            confidence_threshold: Minimum confidence for detections. If None, uses config default.
            device: Device to run inference on ('cpu', 'cuda', '0'). If None, uses config default.
            use_temporal_smoothing: Enable temporal smoothing for video streams
            smoothing_window: Number of frames to use for smoothing
        """
        self.model_path = model_path or ModelConfig.MODEL_WEIGHTS
        self.confidence_threshold = confidence_threshold or ModelConfig.CONFIDENCE_THRESHOLD
        self.device = device or ModelConfig.DEVICE
        self.use_temporal_smoothing = use_temporal_smoothing
        self.smoothing_window = smoothing_window
        
        # Initialize temporal smoothing buffer
        self.count_history = deque(maxlen=smoothing_window)
        
        # Load model
        self.model = None
        self._load_model()
        
        logger.info(f"PassengerDetector initialized with confidence threshold: {self.confidence_threshold}")
    
    def _load_model(self):
        """
        Load the YOLOv5 model.
        
        Raises:
            Exception: If model loading fails
        """
        try:
            logger.info(f"Loading YOLOv5 model from: {self.model_path}")
            
            # Check if custom weights exist, otherwise use pretrained
            if self.model_path.exists():
                self.model = torch.hub.load(
                    'ultralytics/yolov5',
                    'custom',
                    path=str(self.model_path),
                    force_reload=False
                )
                logger.info("Custom model loaded successfully")
            else:
                # Download pretrained model
                logger.info(f"Custom weights not found. Downloading {ModelConfig.MODEL_VARIANT}...")
                self.model = torch.hub.load(
                    'ultralytics/yolov5',
                    ModelConfig.MODEL_VARIANT,
                    pretrained=True
                )
                # Save the model for future use
                torch.save(self.model.state_dict(), self.model_path)
                logger.info(f"Pretrained model downloaded and saved to {self.model_path}")
            
            # Configure model
            self.model.conf = self.confidence_threshold
            self.model.iou = ModelConfig.IOU_THRESHOLD
            self.model.classes = [ModelConfig.PERSON_CLASS_ID]  # Only detect persons
            self.model.max_det = ModelConfig.MAX_DETECTIONS
            
            # Set device
            self.model.to(self.device)
            
            logger.info(f"Model configured and moved to device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect passengers in a single image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
        
        Returns:
            Dictionary containing:
                - count: Number of detected passengers
                - detections: List of detection boxes [x1, y1, x2, y2, confidence]
                - smoothed_count: Temporally smoothed count (if enabled)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Run inference
        results = self.model(image)
        
        # Extract detections
        detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, class]
        
        # Filter for person class only (should already be filtered by model.classes)
        person_detections = detections[detections[:, 5] == ModelConfig.PERSON_CLASS_ID]
        
        # Count passengers
        count = len(person_detections)
        
        # Apply temporal smoothing if enabled
        if self.use_temporal_smoothing:
            self.count_history.append(count)
            smoothed_count = int(np.mean(self.count_history))
        else:
            smoothed_count = count
        
        # Prepare detection data
        detection_boxes = []
        for det in person_detections:
            x1, y1, x2, y2, conf, cls = det
            detection_boxes.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(conf),
                'class': 'person'
            })
        
        return {
            'count': count,
            'smoothed_count': smoothed_count,
            'detections': detection_boxes,
            'raw_results': results
        }
    
    def detect_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Detect passengers in a batch of images.
        
        Args:
            images: List of input images as numpy arrays
        
        Returns:
            List of detection dictionaries (same format as detect())
        """
        results_list = []
        
        for image in images:
            result = self.detect(image)
            results_list.append(result)
        
        return results_list
    
    def annotate_image(
        self,
        image: np.ndarray,
        detection_result: Dict,
        show_confidence: bool = True,
        show_count: bool = True
    ) -> np.ndarray:
        """
        Annotate image with detection results.
        
        Args:
            image: Input image to annotate
            detection_result: Detection result from detect() method
            show_confidence: Whether to show confidence scores on bounding boxes
            show_count: Whether to show total count on image
        
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Draw bounding boxes
        for det in detection_result['detections']:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Draw rectangle
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                VideoConfig.BBOX_COLOR,
                VideoConfig.BBOX_THICKNESS
            )
            
            # Draw confidence score if enabled
            if show_confidence:
                label = f"{conf:.2f}"
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    VideoConfig.BBOX_COLOR,
                    1
                )
        
        # Draw total count if enabled
        if show_count:
            count = detection_result['smoothed_count'] if self.use_temporal_smoothing else detection_result['count']
            count_text = f"Passengers: {count}"
            
            # Add background rectangle for better visibility
            text_size = cv2.getTextSize(
                count_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                VideoConfig.TEXT_SCALE,
                VideoConfig.TEXT_THICKNESS
            )[0]
            
            cv2.rectangle(
                annotated,
                (10, 10),
                (20 + text_size[0], 20 + text_size[1]),
                (0, 0, 0),
                -1
            )
            
            cv2.putText(
                annotated,
                count_text,
                (15, 15 + text_size[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                VideoConfig.TEXT_SCALE,
                VideoConfig.TEXT_COLOR,
                VideoConfig.TEXT_THICKNESS
            )
        
        return annotated
    
    def reset_smoothing(self):
        """Reset temporal smoothing buffer. Useful when switching video sources."""
        self.count_history.clear()
        logger.debug("Temporal smoothing buffer reset")
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_path': str(self.model_path),
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': ModelConfig.IOU_THRESHOLD,
            'device': self.device,
            'temporal_smoothing': self.use_temporal_smoothing,
            'smoothing_window': self.smoothing_window
        }


def test_detector():
    """Test function to verify detector functionality."""
    logger.info("Testing PassengerDetector...")
    
    # Create detector instance
    detector = PassengerDetector()
    
    # Print model info
    print("\nModel Information:")
    print("-" * 50)
    for key, value in detector.get_model_info().items():
        print(f"{key}: {value}")
    
    # Create a dummy image for testing
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Run detection
    result = detector.detect(dummy_image)
    
    print("\nTest Detection Result:")
    print("-" * 50)
    print(f"Detected passengers: {result['count']}")
    print(f"Number of detections: {len(result['detections'])}")
    
    logger.info("Detector test completed successfully")


if __name__ == "__main__":
    test_detector()
