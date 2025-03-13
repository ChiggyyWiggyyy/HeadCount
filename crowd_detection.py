import os
import torch
from shutil import copyfile
from yolov5 import detect

# Define paths
base_dir = '/home/devansh/Rohan'  # Change this to the correct path in your local environment
images_dir = os.path.join(base_dir, 'JPEGImages')  # Directory containing images
labels_dir = os.path.join(base_dir, 'YOLO_Annotations')  # Directory containing YOLO format text files
yolov5_dir = os.path.join(base_dir, 'yolov5')

# Ensure directories exist
assert os.path.exists(images_dir), f"Images directory not found: {images_dir}"
assert os.path.exists(labels_dir), f"Labels directory not found: {labels_dir}"

# Create directories for YOLOv5
train_images_dir = os.path.join(yolov5_dir, 'images/train')
val_images_dir = os.path.join(yolov5_dir, 'images/val')
train_labels_dir = os.path.join(yolov5_dir, 'labels/train')
val_labels_dir = os.path.join(yolov5_dir, 'labels/val')

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Split dataset (80% train, 20% validation)
image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
split_index = int(0.8 * len(image_files))

for i, image_file in enumerate(image_files):
    if i % 1000 == 0:
       print("image copying")
    src_image = os.path.join(images_dir, image_file)
    src_label = os.path.join(labels_dir, image_file.replace('.jpg', '.txt'))

    if i < split_index:
        dest_image = os.path.join(train_images_dir, image_file)
        dest_label = os.path.join(train_labels_dir, image_file.replace('.jpg', '.txt'))
    else:
        dest_image = os.path.join(val_images_dir, image_file)
        dest_label = os.path.join(val_labels_dir, image_file.replace('.jpg', '.txt'))

    copyfile(src_image, dest_image)
    copyfile(src_label, dest_label)

# Create data.yaml
data_yaml = f"""
train: {train_images_dir}
val: {val_images_dir}

nc: 1  # number of classes
names: ['person']  # class names
"""

with open('yolov5/data.yaml', 'w') as f:
    f.write(data_yaml)

# Train YOLOv5
os.system('python yolov5/train.py --img 640 --batch 16 --epochs 50 --data yolov5/data.yaml --weights yolov5s.pt --cache --device 0')

# Run inference
def run_inference(source, weights):
    # Load model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True)

    # Perform inference
    results = model(source)
    return results

# Example of running inference and counting detected heads
source = 'yolov5/images/val'  # Directory of validation images
weights = 'runs/train/exp/weights/best.pt'  # Path to the best model weights

results = run_inference(source, weights)

# Count detected heads (assuming class 0 is 'person')
head_counts = []
for result in results.xyxy:
    head_count = sum(1 for det in result if det[-1] == 0)  # Assuming class 0 is 'person'
    head_counts.append(head_count)

# Print head counts for each image
for i, count in enumerate(head_counts):
    print(f"Image {i}: {count} heads detected")

# Generate message if crowd exceeds threshold
threshold = 50  # Define your threshold for crowd size
if any(count > threshold for count in head_counts):
    print("Warning: Crowd size exceeds threshold!")

