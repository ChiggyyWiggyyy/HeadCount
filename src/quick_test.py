"""
Quick Test Script

This script performs a quick test of the Head Count Project system
using the sample screenshots in the data/input directory.

Author: Head Count Project Team
Date: 2025-12-20
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from passenger_detector import PassengerDetector
from alert_system import AlertSystem
from video_processor import VideoProcessor
from config import INPUT_DIR, OUTPUT_DIR

def main():
    print("=" * 70)
    print("HEAD COUNT PROJECT - QUICK TEST")
    print("=" * 70)
    
    # Create components
    print("\n1. Initializing components...")
    detector = PassengerDetector()
    alert_system = AlertSystem()
    processor = VideoProcessor(detector=detector, alert_system=alert_system)
    
    print("✓ Components initialized successfully")
    
    # Find test images
    print("\n2. Finding test images...")
    test_images = list(INPUT_DIR.glob("Screenshot*.jpg"))
    print(f"✓ Found {len(test_images)} test images")
    
    if not test_images:
        print("\n⚠ No test images found in data/input/")
        print("Please add some images to test.")
        return
    
    # Process images
    print("\n3. Processing images...")
    results = processor.process_batch(INPUT_DIR, OUTPUT_DIR)
    
    print(f"\n✓ Processed {len(results)} images")
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    for i, result in enumerate(results, 1):
        img_name = Path(result['image_path']).name
        count = result['passenger_count']
        print(f"{i}. {img_name}: {count} passengers detected")
    
    total = sum(r['passenger_count'] for r in results)
    avg = total / len(results) if results else 0
    
    print(f"\nTotal passengers: {total}")
    print(f"Average per image: {avg:.2f}")
    print(f"\nAnnotated images saved to: {OUTPUT_DIR}")
    
    # Alert summary
    alert_history = alert_system.get_alert_history()
    if alert_history:
        print(f"\n⚠ Alerts triggered: {len(alert_history)}")
        for alert in alert_history:
            print(f"  - [{alert['alert_level'].upper()}] {alert['message']}")
    
    print("\n" + "=" * 70)
    print("✓ Test completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
