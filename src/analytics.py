"""
Analytics Module

This module provides data analytics and reporting capabilities for the
passenger counting system. It tracks statistics, generates reports, and
exports data for further analysis.

Key Features:
- Real-time statistics tracking
- Historical data analysis
- Peak time detection
- CSV export for external analysis
- Occupancy trend analysis

Author: Head Count Project Team
Date: 2025-12-20
"""

import csv
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import logging
import json
import numpy as np

from config import AnalyticsConfig, LOGS_DIR

# Set up logging
logger = logging.getLogger(__name__)


class Analytics:
    """
    Analytics class for tracking and analyzing passenger count data.
    
    This class collects passenger count data over time and provides
    statistical analysis and reporting capabilities.
    """
    
    def __init__(
        self,
        enable_logging: bool = None,
        log_file: Optional[Path] = None
    ):
        """
        Initialize the analytics system.
        
        Args:
            enable_logging: Enable data logging to file
            log_file: Path to analytics log file (uses config default if None)
        """
        self.enable_logging = enable_logging if enable_logging is not None else AnalyticsConfig.ENABLE_ANALYTICS
        self.log_file = log_file or AnalyticsConfig.ANALYTICS_LOG_FILE
        
        # Data storage
        self.data_points: List[Dict] = []
        
        # Initialize file logging
        if self.enable_logging:
            self._init_logging()
        
        logger.info("Analytics system initialized")
    
    def _init_logging(self):
        """Initialize CSV logging file."""
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create file with header if it doesn't exist
            if not self.log_file.exists():
                with open(self.log_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp',
                        'passenger_count',
                        'smoothed_count',
                        'num_detections',
                        'source',
                        'frame_number'
                    ])
            
            logger.info(f"Analytics logging initialized: {self.log_file}")
        except Exception as e:
            logger.error(f"Failed to initialize analytics logging: {e}")
            self.enable_logging = False
    
    def record_data_point(
        self,
        passenger_count: int,
        smoothed_count: int = None,
        num_detections: int = None,
        source: str = "unknown",
        frame_number: int = None,
        metadata: Dict = None
    ):
        """
        Record a data point for analysis.
        
        Args:
            passenger_count: Raw passenger count
            smoothed_count: Temporally smoothed count
            num_detections: Number of individual detections
            source: Source identifier (e.g., camera ID, video file name)
            frame_number: Frame number in video stream
            metadata: Additional metadata dictionary
        """
        timestamp = datetime.now()
        
        data_point = {
            'timestamp': timestamp,
            'passenger_count': passenger_count,
            'smoothed_count': smoothed_count or passenger_count,
            'num_detections': num_detections or passenger_count,
            'source': source,
            'frame_number': frame_number,
            'metadata': metadata or {}
        }
        
        self.data_points.append(data_point)
        
        # Log to file if enabled
        if self.enable_logging:
            self._log_to_file(data_point)
    
    def _log_to_file(self, data_point: Dict):
        """Log data point to CSV file."""
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    data_point['timestamp'].isoformat(),
                    data_point['passenger_count'],
                    data_point['smoothed_count'],
                    data_point['num_detections'],
                    data_point['source'],
                    data_point['frame_number']
                ])
        except Exception as e:
            logger.error(f"Failed to log data point: {e}")
    
    def get_statistics(self) -> Dict:
        """
        Calculate statistics from collected data.
        
        Returns:
            Dictionary with statistical information
        """
        if not self.data_points:
            return {
                'total_data_points': 0,
                'message': 'No data points recorded'
            }
        
        counts = [dp['passenger_count'] for dp in self.data_points]
        smoothed_counts = [dp['smoothed_count'] for dp in self.data_points]
        
        stats = {
            'total_data_points': len(self.data_points),
            'time_range': {
                'start': self.data_points[0]['timestamp'].isoformat(),
                'end': self.data_points[-1]['timestamp'].isoformat(),
                'duration_seconds': (self.data_points[-1]['timestamp'] - self.data_points[0]['timestamp']).total_seconds()
            },
            'passenger_counts': {
                'min': min(counts),
                'max': max(counts),
                'mean': np.mean(counts),
                'median': np.median(counts),
                'std': np.std(counts)
            },
            'smoothed_counts': {
                'min': min(smoothed_counts),
                'max': max(smoothed_counts),
                'mean': np.mean(smoothed_counts),
                'median': np.median(smoothed_counts),
                'std': np.std(smoothed_counts)
            }
        }
        
        return stats
    
    def get_peak_times(self, top_n: int = 5) -> List[Dict]:
        """
        Get peak occupancy times.
        
        Args:
            top_n: Number of top peaks to return
        
        Returns:
            List of peak time data points
        """
        if not self.data_points:
            return []
        
        # Sort by passenger count
        sorted_points = sorted(
            self.data_points,
            key=lambda x: x['passenger_count'],
            reverse=True
        )
        
        return sorted_points[:top_n]
    
    def get_occupancy_trend(self, window_size: int = 10) -> List[Dict]:
        """
        Calculate occupancy trend using moving average.
        
        Args:
            window_size: Size of moving average window
        
        Returns:
            List of trend data points
        """
        if len(self.data_points) < window_size:
            return []
        
        trend = []
        for i in range(len(self.data_points) - window_size + 1):
            window = self.data_points[i:i + window_size]
            avg_count = np.mean([dp['passenger_count'] for dp in window])
            
            trend.append({
                'timestamp': window[-1]['timestamp'],
                'moving_average': avg_count,
                'window_size': window_size
            })
        
        return trend
    
    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate a comprehensive analytics report.
        
        Args:
            output_path: Path to save report (optional)
        
        Returns:
            Report as formatted string
        """
        stats = self.get_statistics()
        peaks = self.get_peak_times(5)
        
        report_lines = [
            "=" * 70,
            "PASSENGER COUNT ANALYTICS REPORT",
            "=" * 70,
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nTotal Data Points: {stats.get('total_data_points', 0)}",
        ]
        
        if stats.get('total_data_points', 0) > 0:
            time_range = stats['time_range']
            report_lines.extend([
                f"\nTime Range:",
                f"  Start: {time_range['start']}",
                f"  End: {time_range['end']}",
                f"  Duration: {time_range['duration_seconds']:.2f} seconds",
                f"\nPassenger Count Statistics:",
                f"  Minimum: {stats['passenger_counts']['min']}",
                f"  Maximum: {stats['passenger_counts']['max']}",
                f"  Mean: {stats['passenger_counts']['mean']:.2f}",
                f"  Median: {stats['passenger_counts']['median']:.2f}",
                f"  Std Dev: {stats['passenger_counts']['std']:.2f}",
            ])
            
            if peaks:
                report_lines.append(f"\nTop 5 Peak Occupancy Times:")
                for i, peak in enumerate(peaks, 1):
                    report_lines.append(
                        f"  {i}. {peak['timestamp'].strftime('%H:%M:%S')} - "
                        f"{peak['passenger_count']} passengers (Source: {peak['source']})"
                    )
        
        report_lines.append("\n" + "=" * 70)
        
        report = "\n".join(report_lines)
        
        # Save to file if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to: {output_path}")
        
        return report
    
    def export_to_csv(self, output_path: Path):
        """
        Export all data points to CSV file.
        
        Args:
            output_path: Path for output CSV file
        """
        if not self.data_points:
            logger.warning("No data points to export")
            return
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'timestamp',
                'passenger_count',
                'smoothed_count',
                'num_detections',
                'source',
                'frame_number'
            ])
            
            # Write data
            for dp in self.data_points:
                writer.writerow([
                    dp['timestamp'].isoformat(),
                    dp['passenger_count'],
                    dp['smoothed_count'],
                    dp['num_detections'],
                    dp['source'],
                    dp['frame_number']
                ])
        
        logger.info(f"Data exported to: {output_path}")
    
    def export_to_json(self, output_path: Path):
        """
        Export all data points to JSON file.
        
        Args:
            output_path: Path for output JSON file
        """
        if not self.data_points:
            logger.warning("No data points to export")
            return
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert datetime objects to ISO format strings
        export_data = []
        for dp in self.data_points:
            dp_copy = dp.copy()
            dp_copy['timestamp'] = dp['timestamp'].isoformat()
            export_data.append(dp_copy)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Data exported to: {output_path}")
    
    def clear_data(self):
        """Clear all collected data points."""
        self.data_points.clear()
        logger.info("Analytics data cleared")
    
    def get_data_points(self, limit: int = None) -> List[Dict]:
        """
        Get recorded data points.
        
        Args:
            limit: Maximum number of recent points to return (None for all)
        
        Returns:
            List of data point dictionaries
        """
        if limit is None:
            return self.data_points.copy()
        return self.data_points[-limit:]


def test_analytics():
    """Test function for analytics module."""
    print("Testing Analytics...")
    print("=" * 60)
    
    # Create analytics instance
    analytics = Analytics()
    
    # Simulate some data points
    import random
    for i in range(20):
        count = random.randint(10, 80)
        analytics.record_data_point(
            passenger_count=count,
            smoothed_count=count,
            source="test_camera",
            frame_number=i
        )
    
    # Get statistics
    stats = analytics.get_statistics()
    print("\nStatistics:")
    print(json.dumps(stats, indent=2, default=str))
    
    # Get peak times
    peaks = analytics.get_peak_times(3)
    print("\nTop 3 Peaks:")
    for i, peak in enumerate(peaks, 1):
        print(f"  {i}. {peak['passenger_count']} passengers at {peak['timestamp']}")
    
    # Generate report
    print("\n" + analytics.generate_report())
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_analytics()
