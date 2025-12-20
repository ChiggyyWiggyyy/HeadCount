"""
Main Application Entry Point

This is the main application for the Head Count Project - a passenger counting
system for public transport vehicles. It provides a command-line interface for
processing videos, images, and live camera feeds.

Usage Examples:
    # Process a video file
    python main.py --video path/to/video.mp4
    
    # Process from webcam
    python main.py --camera 0
    
    # Process images in a directory
    python main.py --images path/to/images/
    
    # Process with custom thresholds
    python main.py --video video.mp4 --warning 60 --critical 80

Author: Head Count Project Team
Date: 2025-12-20
"""

import argparse
import sys
from pathlib import Path
import logging
import json

from config import (
    ModelConfig, AlertConfig, LoggingConfig,
    get_config_summary, PROJECT_ROOT, OUTPUT_DIR
)
from passenger_detector import PassengerDetector
from alert_system import AlertSystem
from video_processor import VideoProcessor
from analytics import Analytics

# Set up logging
logging.basicConfig(
    level=getattr(logging, LoggingConfig.LOG_LEVEL),
    format=LoggingConfig.LOG_FORMAT,
    datefmt=LoggingConfig.DATE_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LoggingConfig.LOG_FILE)
    ] if LoggingConfig.FILE_LOGGING else [logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print application banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    HEAD COUNT PROJECT                        ║
    ║          Intelligent Passenger Counting System               ║
    ║              for Public Transport Vehicles                   ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def process_video_mode(args):
    """
    Process video file or camera stream.
    
    Args:
        args: Command-line arguments
    """
    logger.info("Starting video processing mode")
    
    # Create components
    detector = PassengerDetector(
        confidence_threshold=args.confidence,
        device=args.device
    )
    
    alert_system = AlertSystem(
        warning_threshold=args.warning,
        critical_threshold=args.critical
    )
    
    analytics = Analytics()
    
    processor = VideoProcessor(
        detector=detector,
        alert_system=alert_system,
        frame_skip=args.frame_skip,
        save_output=args.save_output,
        show_preview=args.preview
    )
    
    # Callback to record analytics
    def analytics_callback(frame_num, detection_result):
        analytics.record_data_point(
            passenger_count=detection_result['count'],
            smoothed_count=detection_result['smoothed_count'],
            num_detections=len(detection_result['detections']),
            source=str(args.video or args.camera),
            frame_number=frame_num
        )
    
    # Process video
    video_source = args.video if args.video else args.camera
    results = processor.process_video(
        video_source=video_source,
        output_path=Path(args.output) if args.output else None,
        callback=analytics_callback
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("PROCESSING RESULTS")
    print("=" * 70)
    print(f"Source: {results['source']}")
    print(f"Type: {results['source_type']}")
    print(f"Output: {results['output_path']}")
    print(f"\nStatistics:")
    print(f"  Total frames: {results['statistics']['total_frames']}")
    print(f"  Processed frames: {results['statistics']['processed_frames']}")
    print(f"  Skipped frames: {results['statistics']['skipped_frames']}")
    print(f"  Processing time: {results['statistics']['total_time']:.2f}s")
    print(f"  Average FPS: {results['statistics']['avg_fps']:.2f}")
    print(f"\nPassenger Counts:")
    print(f"  Maximum: {results['max_passengers']}")
    print(f"  Average: {results['avg_passengers']:.2f}")
    print(f"\nAlerts triggered: {len(results['alert_history'])}")
    
    if results['alert_history']:
        print("\nAlert History:")
        for alert in results['alert_history']:
            print(f"  [{alert['alert_level'].upper()}] {alert['message']}")
    
    # Generate analytics report
    if args.report:
        report_path = OUTPUT_DIR / "analytics_report.txt"
        report = analytics.generate_report(report_path)
        print(f"\nAnalytics report saved to: {report_path}")
    
    print("=" * 70)


def process_image_mode(args):
    """
    Process single image or batch of images.
    
    Args:
        args: Command-line arguments
    """
    logger.info("Starting image processing mode")
    
    # Create components
    detector = PassengerDetector(
        confidence_threshold=args.confidence,
        device=args.device
    )
    
    alert_system = AlertSystem(
        warning_threshold=args.warning,
        critical_threshold=args.critical
    )
    
    processor = VideoProcessor(
        detector=detector,
        alert_system=alert_system,
        save_output=args.save_output
    )
    
    input_path = Path(args.images)
    
    if input_path.is_file():
        # Process single image
        result = processor.process_image(
            image_path=input_path,
            output_path=Path(args.output) if args.output else None
        )
        
        print("\n" + "=" * 70)
        print("IMAGE PROCESSING RESULT")
        print("=" * 70)
        print(f"Input: {result['image_path']}")
        print(f"Output: {result['output_path']}")
        print(f"Passengers detected: {result['passenger_count']}")
        print(f"Number of detections: {len(result['detections'])}")
        
        if result['alert']:
            print(f"\nAlert: [{result['alert']['alert_level'].upper()}] {result['alert']['message']}")
        
        print("=" * 70)
    
    elif input_path.is_dir():
        # Process batch of images
        output_dir = Path(args.output) if args.output else OUTPUT_DIR
        results = processor.process_batch(
            input_dir=input_path,
            output_dir=output_dir
        )
        
        print("\n" + "=" * 70)
        print("BATCH PROCESSING RESULTS")
        print("=" * 70)
        print(f"Input directory: {input_path}")
        print(f"Output directory: {output_dir}")
        print(f"Images processed: {len(results)}")
        
        total_passengers = sum(r['passenger_count'] for r in results)
        avg_passengers = total_passengers / len(results) if results else 0
        
        print(f"\nTotal passengers detected: {total_passengers}")
        print(f"Average per image: {avg_passengers:.2f}")
        
        # Show top 5 images by passenger count
        sorted_results = sorted(results, key=lambda x: x['passenger_count'], reverse=True)
        print("\nTop 5 images by passenger count:")
        for i, result in enumerate(sorted_results[:5], 1):
            print(f"  {i}. {Path(result['image_path']).name}: {result['passenger_count']} passengers")
        
        print("=" * 70)
    
    else:
        logger.error(f"Invalid input path: {input_path}")
        sys.exit(1)


def show_config():
    """Display current configuration."""
    print("\n" + "=" * 70)
    print("CURRENT CONFIGURATION")
    print("=" * 70)
    config = get_config_summary()
    print(json.dumps(config, indent=2))
    print("=" * 70)


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Head Count Project - Passenger Counting System for Public Transport",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process video file:
    python main.py --video input.mp4
  
  Process from webcam:
    python main.py --camera 0 --preview
  
  Process images:
    python main.py --images ./photos/
  
  Custom thresholds:
    python main.py --video bus.mp4 --warning 60 --critical 80
        """
    )
    
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('--video', type=str, help='Path to video file')
    input_group.add_argument('--camera', type=int, help='Camera index (0 for default camera)')
    input_group.add_argument('--images', type=str, help='Path to image file or directory')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output path for processed video/images')
    parser.add_argument('--no-save', dest='save_output', action='store_false',
                       help='Disable saving output video/images')
    parser.add_argument('--preview', action='store_true',
                       help='Show live preview window (video mode only)')
    
    # Detection parameters
    parser.add_argument('--confidence', type=float, default=ModelConfig.CONFIDENCE_THRESHOLD,
                       help=f'Detection confidence threshold (default: {ModelConfig.CONFIDENCE_THRESHOLD})')
    parser.add_argument('--device', type=str, default=ModelConfig.DEVICE,
                       help=f'Device for inference: cpu, cuda, 0, 1, etc. (default: {ModelConfig.DEVICE})')
    parser.add_argument('--frame-skip', type=int, default=1,
                       help='Process every Nth frame (default: 1, process all frames)')
    
    # Alert thresholds
    parser.add_argument('--warning', type=int, default=AlertConfig.WARNING_THRESHOLD,
                       help=f'Warning threshold for passenger count (default: {AlertConfig.WARNING_THRESHOLD})')
    parser.add_argument('--critical', type=int, default=AlertConfig.CRITICAL_THRESHOLD,
                       help=f'Critical threshold for passenger count (default: {AlertConfig.CRITICAL_THRESHOLD})')
    
    # Analytics
    parser.add_argument('--report', action='store_true',
                       help='Generate analytics report after processing')
    
    # Utility options
    parser.add_argument('--config', action='store_true',
                       help='Show current configuration and exit')
    parser.add_argument('--version', action='version', version='Head Count Project v1.0')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Show config if requested
    if args.config:
        show_config()
        return
    
    # Check if any input source is provided
    if not (args.video or args.camera is not None or args.images):
        parser.print_help()
        print("\nError: Please specify an input source (--video, --camera, or --images)")
        sys.exit(1)
    
    try:
        # Route to appropriate processing mode
        if args.video or args.camera is not None:
            process_video_mode(args)
        elif args.images:
            process_image_mode(args)
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        print("\n\nProcessing interrupted by user.")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
