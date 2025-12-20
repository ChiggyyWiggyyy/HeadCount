"""
Model Training Script

This script provides functionality for training a custom YOLOv5 model
on a custom dataset for passenger detection. Use this if you want to
fine-tune the model on your own data.

Usage:
    python train_model.py --data path/to/dataset --epochs 50

Author: Head Count Project Team
Date: 2025-12-20
"""

import argparse
import sys
from pathlib import Path
import logging
import yaml
import shutil

from config import TrainingConfig, MODELS_DIR, DATA_DIR, PROJECT_ROOT

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_dataset(dataset_path: Path, output_path: Path, train_split: float = 0.8):
    """
    Prepare dataset in YOLOv5 format.
    
    Expected input structure:
        dataset_path/
            images/
                image1.jpg
                image2.jpg
                ...
            labels/
                image1.txt
                image2.txt
                ...
    
    Args:
        dataset_path: Path to input dataset
        output_path: Path for prepared dataset
        train_split: Ratio for train/val split
    """
    logger.info(f"Preparing dataset from {dataset_path}")
    
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    
    if not images_dir.exists() or not labels_dir.exists():
        raise ValueError(f"Dataset must contain 'images' and 'labels' directories")
    
    # Create output structure
    train_images = output_path / "images" / "train"
    val_images = output_path / "images" / "val"
    train_labels = output_path / "labels" / "train"
    val_labels = output_path / "labels" / "val"
    
    for dir_path in [train_images, val_images, train_labels, val_labels]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    total_images = len(image_files)
    split_idx = int(total_images * train_split)
    
    logger.info(f"Found {total_images} images. Split: {split_idx} train, {total_images - split_idx} val")
    
    # Split and copy files
    for i, img_path in enumerate(image_files):
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            logger.warning(f"Label not found for {img_path.name}, skipping")
            continue
        
        # Determine destination
        if i < split_idx:
            dest_img_dir = train_images
            dest_label_dir = train_labels
        else:
            dest_img_dir = val_images
            dest_label_dir = val_labels
        
        # Copy files
        shutil.copy(img_path, dest_img_dir / img_path.name)
        shutil.copy(label_path, dest_label_dir / label_path.name)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{total_images} images")
    
    logger.info("Dataset preparation completed")
    return output_path


def create_data_yaml(dataset_path: Path, num_classes: int = 1, class_names: list = None):
    """
    Create data.yaml configuration file for YOLOv5 training.
    
    Args:
        dataset_path: Path to prepared dataset
        num_classes: Number of classes
        class_names: List of class names
    
    Returns:
        Path to created data.yaml file
    """
    if class_names is None:
        class_names = ['person']
    
    data_yaml = {
        'path': str(dataset_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': num_classes,
        'names': class_names
    }
    
    yaml_path = dataset_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    logger.info(f"Created data.yaml at {yaml_path}")
    return yaml_path


def train_model(
    data_yaml: Path,
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    weights: str = 'yolov5s.pt',
    device: str = '0',
    project: Path = None,
    name: str = 'passenger_detection'
):
    """
    Train YOLOv5 model.
    
    Args:
        data_yaml: Path to data.yaml configuration
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        weights: Pretrained weights to start from
        device: Device to train on ('cpu', '0', '1', etc.)
        project: Project directory for saving results
        name: Name for this training run
    """
    logger.info("Starting model training...")
    
    # Import YOLOv5 train module
    try:
        import torch
        
        # Clone YOLOv5 if not present
        yolov5_path = PROJECT_ROOT / "yolov5"
        if not yolov5_path.exists():
            logger.info("Cloning YOLOv5 repository...")
            import os
            os.system(f"git clone https://github.com/ultralytics/yolov5 {yolov5_path}")
        
        # Add to path
        sys.path.insert(0, str(yolov5_path))
        
        # Import train
        from train import train as yolo_train, parse_opt
        
        # Prepare arguments
        opt = parse_opt(known=True)
        opt.data = str(data_yaml)
        opt.epochs = epochs
        opt.batch_size = batch_size
        opt.imgsz = img_size
        opt.weights = weights
        opt.device = device
        opt.project = str(project or MODELS_DIR)
        opt.name = name
        opt.cache = True
        
        # Train
        logger.info(f"Training with: epochs={epochs}, batch={batch_size}, img_size={img_size}")
        yolo_train(opt)
        
        logger.info("Training completed!")
        logger.info(f"Results saved to: {opt.project}/{opt.name}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.info("Attempting alternative training method using command line...")
        
        # Fallback to command line
        import os
        cmd = (
            f"python -m torch.distributed.run --nproc_per_node 1 "
            f"yolov5/train.py --data {data_yaml} --epochs {epochs} "
            f"--batch {batch_size} --img {img_size} --weights {weights} "
            f"--device {device} --project {project or MODELS_DIR} --name {name} --cache"
        )
        
        logger.info(f"Running: {cmd}")
        os.system(cmd)


def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(
        description="Train custom YOLOv5 model for passenger detection"
    )
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset directory (must contain images/ and labels/)')
    parser.add_argument('--epochs', type=int, default=TrainingConfig.EPOCHS,
                       help=f'Number of training epochs (default: {TrainingConfig.EPOCHS})')
    parser.add_argument('--batch', type=int, default=TrainingConfig.BATCH_SIZE,
                       help=f'Batch size (default: {TrainingConfig.BATCH_SIZE})')
    parser.add_argument('--img-size', type=int, default=TrainingConfig.IMG_SIZE,
                       help=f'Image size (default: {TrainingConfig.IMG_SIZE})')
    parser.add_argument('--weights', type=str, default='yolov5s.pt',
                       help='Pretrained weights to start from (default: yolov5s.pt)')
    parser.add_argument('--device', type=str, default='0',
                       help='Device to train on: cpu, 0, 1, etc. (default: 0)')
    parser.add_argument('--name', type=str, default='passenger_detection',
                       help='Name for this training run (default: passenger_detection)')
    parser.add_argument('--split', type=float, default=TrainingConfig.TRAIN_VAL_SPLIT,
                       help=f'Train/val split ratio (default: {TrainingConfig.TRAIN_VAL_SPLIT})')
    parser.add_argument('--skip-prep', action='store_true',
                       help='Skip dataset preparation (use if already prepared)')
    
    args = parser.parse_args()
    
    try:
        dataset_path = Path(args.data)
        
        if not dataset_path.exists():
            logger.error(f"Dataset path not found: {dataset_path}")
            sys.exit(1)
        
        # Prepare dataset
        if not args.skip_prep:
            prepared_path = DATA_DIR / "prepared_dataset"
            prepare_dataset(dataset_path, prepared_path, args.split)
        else:
            prepared_path = dataset_path
            logger.info(f"Skipping dataset preparation, using: {prepared_path}")
        
        # Create data.yaml
        data_yaml = create_data_yaml(prepared_path)
        
        # Train model
        train_model(
            data_yaml=data_yaml,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            weights=args.weights,
            device=args.device,
            name=args.name
        )
        
        logger.info("Training process completed successfully!")
        logger.info(f"Trained model weights can be found in: {MODELS_DIR}/{args.name}/weights/best.pt")
        logger.info("You can use these weights with the main application using --weights flag")
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
