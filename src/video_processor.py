"""
Video Processing Module

This module handles video stream processing for passenger detection.
It supports both live camera feeds and video files, providing real-time
passenger counting with optional output video generation.

Key Features:
- Live camera feed processing
- Video file processing
- Frame-by-frame analysis
- Output video generation with annotations
- Batch processing support
- Performance optimization with frame skipping

Author: Head Count Project Team
Date: 2025-12-20
"""

import cv2
import numpy as np
from typing import Optional, Callable, Dict, List
from pathlib import Path
import logging
from datetime import datetime
import time

from config import VideoConfig, OUTPUT_DIR
from passenger_detector import PassengerDetector
from alert_system import AlertSystem

# Set up logging
logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Video processing class for passenger detection in video streams.
    
    This class handles video input (file or camera), processes frames through
    the passenger detector, and optionally generates annotated output videos.
    """
    
    def __init__(
        self,
        detector: Optional[PassengerDetector] = None,
        alert_system: Optional[AlertSystem] = None,
        frame_skip: int = None,
        save_output: bool = None,
        show_preview: bool = False
    ):
        """
        Initialize the video processor.
        
        Args:
            detector: PassengerDetector instance (creates new if None)
            alert_system: AlertSystem instance (creates new if None)
            frame_skip: Process every Nth frame (None uses config default)
            save_output: Save annotated output video (None uses config default)
            show_preview: Show live preview window during processing
        """
        self.detector = detector or PassengerDetector()
        self.alert_system = alert_system or AlertSystem()
        self.frame_skip = frame_skip if frame_skip is not None else VideoConfig.FRAME_SKIP
        self.save_output = save_output if save_output is not None else VideoConfig.SAVE_OUTPUT_VIDEO
        self.show_preview = show_preview
        
        # Initialize CLAHE if enabled (this would fix the ghost lag)
        self.clahe = None
        if VideoConfig.ENABLE_CLAHE:
            self.clahe = cv2.createCLAHE(
                clipLimit=VideoConfig.CLAHE_CLIP_LIMIT,
                tileGridSize=VideoConfig.CLAHE_GRID_SIZE
            )
            logger.info(f"CLAHE preprocessing enabled (clip={VideoConfig.CLAHE_CLIP_LIMIT}, grid={VideoConfig.CLAHE_GRID_SIZE})")
        
        # Processing statistics
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'skipped_frames': 0,
            'total_time': 0.0,
            'avg_fps': 0.0
        }
        
        logger.info(f"VideoProcessor initialized (frame_skip={self.frame_skip}, save_output={self.save_output})")
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to image.
        Converts to LAB color space, applies CLAHE to L-channel, and converts back.
        """
        if self.clahe is None:
            return image
            
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L-channel
            cl = self.clahe.apply(l)
            
            # Merge channels
            limg = cv2.merge((cl, a, b))
            
            # Convert back to BGR
            enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            return enhanced_image
        except Exception as e:
            logger.warning(f"CLAHE application failed: {e}")
            return image
    
    def calculate_brightness(self, image: np.ndarray) -> float:
        """
        Calculate average brightness of image (0.0 to 1.0).
        Uses HSV V-channel average.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # V channel is at index 2, range 0-255
        avg_brightness = np.mean(hsv[:, :, 2]) / 255.0
        return avg_brightness

    def apply_gamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply gamma correction to image.
        gamma < 1.0 makes image darker (good for overexposed images)
        gamma > 1.0 makes image brighter
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                        for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def process_video(
        self,
        video_source: str,
        output_path: Optional[Path] = None,
        callback: Optional[Callable[[int, Dict], None]] = None
    ) -> Dict:
        """
        Process a video file or camera stream.
        
        Args:
            video_source: Path to video file or camera index (0 for default camera)
            output_path: Path for output video (auto-generated if None and save_output=True)
            callback: Optional callback function called after each frame processing
                     Signature: callback(frame_number: int, detection_result: Dict)
        
        Returns:
            Dictionary with processing statistics and results
        """
        # Open video source
        if isinstance(video_source, int) or video_source.isdigit():
            cap = cv2.VideoCapture(int(video_source))
            source_type = "camera"
            logger.info(f"Opening camera {video_source}")
        else:
            cap = cv2.VideoCapture(str(video_source))
            source_type = "file"
            logger.info(f"Opening video file: {video_source}")
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video source: {video_source}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or VideoConfig.OUTPUT_FPS
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if source_type == "file" else -1
        
        logger.info(f"Video properties: {width}x{height} @ {fps} FPS, Total frames: {total_frames}")
        
        # Initialize video writer if saving output
        out = None
        if self.save_output:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = OUTPUT_DIR / f"output_{timestamp}.mp4"
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*VideoConfig.OUTPUT_CODEC)
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            logger.info(f"Output video will be saved to: {output_path}")
        
        # Reset statistics
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'skipped_frames': 0,
            'total_time': 0.0,
            'avg_fps': 0.0,
            'passenger_counts': []
        }
        
        # Reset detector smoothing for new video
        self.detector.reset_smoothing()
        
        # Processing loop
        frame_number = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_number += 1
                self.stats['total_frames'] = frame_number
                
                # Frame skipping logic
                if frame_number % (self.frame_skip + 1) != 0:
                    self.stats['skipped_frames'] += 1
                    continue
                
                # Process frame
                frame_start = time.time()
                
                # Apply CLAHE preprocessing if enabled
                processed_frame = self.apply_clahe(frame)
                
                # Apply Gamma Correction if frame is too bright
                if VideoConfig.ENABLE_GAMMA_CORRECTION:
                    brightness = self.calculate_brightness(processed_frame)
                    if brightness > VideoConfig.BRIGHTNESS_THRESHOLD:
                        processed_frame = self.apply_gamma(processed_frame, VideoConfig.GAMMA_VALUE)
                        # Only log occasionally to avoid spam
                        if frame_number % 30 == 0:
                            logger.debug(f"High brightness ({brightness:.2f}) detected - Gamma correction applied")
                
                detection_result = self.detector.detect(processed_frame)
                frame_time = time.time() - frame_start
                
                self.stats['processed_frames'] += 1
                passenger_count = detection_result['smoothed_count']
                self.stats['passenger_counts'].append(passenger_count)
                
                # Check for alerts
                self.alert_system.check_and_alert(passenger_count)
                
                # Annotate frame
                annotated_frame = self.detector.annotate_image(frame, detection_result)
                
                # Add FPS counter
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                cv2.putText(
                    annotated_frame,
                    f"FPS: {current_fps:.1f}",
                    (width - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # Save to output video
                if out is not None:
                    out.write(annotated_frame)
                
                # Show preview if enabled
                if self.show_preview:
                    cv2.imshow('Passenger Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Preview window closed by user")
                        break
                
                # Call callback if provided
                if callback:
                    callback(frame_number, detection_result)
                
                # Progress logging
                if frame_number % 100 == 0:
                    logger.info(f"Processed frame {frame_number}/{total_frames if total_frames > 0 else '?'} "
                               f"- Passengers: {passenger_count}")
        
        finally:
            # Cleanup
            total_time = time.time() - start_time
            self.stats['total_time'] = total_time
            self.stats['avg_fps'] = self.stats['processed_frames'] / total_time if total_time > 0 else 0
            
            cap.release()
            if out is not None:
                out.release()
            if self.show_preview:
                cv2.destroyAllWindows()
            
            logger.info(f"Video processing completed in {total_time:.2f}s "
                       f"(avg {self.stats['avg_fps']:.2f} FPS)")
        
        # Compile results
        results = {
            'source': str(video_source),
            'source_type': source_type,
            'output_path': str(output_path) if output_path else None,
            'statistics': self.stats,
            'max_passengers': max(self.stats['passenger_counts']) if self.stats['passenger_counts'] else 0,
            'avg_passengers': np.mean(self.stats['passenger_counts']) if self.stats['passenger_counts'] else 0,
            'alert_history': self.alert_system.get_alert_history()
        }
        
        return results
    
    def process_image(
        self,
        image_path: Path,
        output_path: Optional[Path] = None
    ) -> Dict:
        """
        Process a single image.
        
        Args:
            image_path: Path to input image
            output_path: Path for output image (auto-generated if None)
        
        Returns:
            Dictionary with detection results
        """
        logger.info(f"Processing image: {image_path}")
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
            
        # Apply CLAHE preprocessing if enabled
        processed_image = self.apply_clahe(image)
        
        # Apply Gamma Correction if image is too bright
        if VideoConfig.ENABLE_GAMMA_CORRECTION:
            brightness = self.calculate_brightness(processed_image)
            if brightness > VideoConfig.BRIGHTNESS_THRESHOLD:
                processed_image = self.apply_gamma(processed_image, VideoConfig.GAMMA_VALUE)
                logger.info(f"High brightness ({brightness:.2f}) detected - Gamma correction applied")
        
        # Detect passengers
        detection_result = self.detector.detect(processed_image)
        passenger_count = detection_result['count']
        
        # Check for alerts
        alert_info = self.alert_system.check_and_alert(passenger_count)
        
        # Annotate image
        annotated_image = self.detector.annotate_image(image, detection_result)
        
        # Save output if requested
        if output_path is None and self.save_output:
            output_path = OUTPUT_DIR / f"annotated_{image_path.name}"
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated_image)
            logger.info(f"Annotated image saved to: {output_path}")
        
        return {
            'image_path': str(image_path),
            'output_path': str(output_path) if output_path else None,
            'passenger_count': passenger_count,
            'detections': detection_result['detections'],
            'alert': alert_info
        }
    
    def process_batch(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        extensions: List[str] = None
    ) -> List[Dict]:
        """
        Process a batch of images from a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory for output images (uses OUTPUT_DIR if None)
            extensions: List of file extensions to process (default: ['.jpg', '.jpeg', '.png'])
        
        Returns:
            List of result dictionaries for each processed image
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        output_dir = output_dir or OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} images to process in {input_dir}")
        
        # Process each image
        results = []
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"Processing image {i}/{len(image_files)}: {image_path.name}")
            
            output_path = output_dir / f"annotated_{image_path.name}"
            result = self.process_image(image_path, output_path)
            results.append(result)
        
        logger.info(f"Batch processing completed: {len(results)} images processed")
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        return self.stats.copy()


def test_video_processor():
    """Test function for video processor."""
    print("Testing VideoProcessor...")
    print("=" * 60)
    
    # Create processor
    processor = VideoProcessor(show_preview=False)
    
    # Create a test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_path = OUTPUT_DIR / "test_image.jpg"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(test_path), test_image)
    
    # Process test image
    result = processor.process_image(test_path)
    
    print("\nTest Image Processing Result:")
    print(f"  Passengers detected: {result['passenger_count']}")
    print(f"  Output saved to: {result['output_path']}")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_video_processor()
