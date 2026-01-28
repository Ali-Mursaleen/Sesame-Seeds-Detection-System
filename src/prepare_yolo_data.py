#!/usr/bin/env python3
"""
Prepare YOLO format dataset from sesame seed images
Handles multi-seed images of same category
"""

import os
import cv2
import numpy as np
import shutil
import yaml
from pathlib import Path
import random
import sys

# Try to import tqdm with error handling
try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm

class YOLODataPreparer:
    def __init__(self):
        self.base_dir = Path(".")
        self.data_dir = self.base_dir / "data"  # Original 60 images
        self.aug_dir = self.base_dir / "Augmented_Data"  # Your augmented data
        self.yolo_dir = self.base_dir / "yolov8"
        
        # YOLO class mapping
        self.classes = ['healthy', 'black', 'rain_damaged']
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Map original folder names to YOLO classes
        self.folder_to_class = {
            'Healthy': 'healthy',
            'Black': 'black', 
            'Rain Damage': 'rain_damaged'
        }
        
    def auto_detect_seeds(self, image):
        """
        Automatically detect individual seeds in an image using contour detection
        Returns list of [x_center, y_center, width, height] for each seed
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        img_height, img_width = image.shape[:2]
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < 50:
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio
            aspect_ratio = w / h
            if aspect_ratio > 3 or aspect_ratio < 0.33:
                continue
                
            # Calculate center
            x_center = x + w / 2
            y_center = y + h / 2
            
            # Add padding
            w_padded = w * 1.2
            h_padded = h * 1.2
            
            # Ensure box stays within bounds
            x_center = max(w_padded/2, min(img_width - w_padded/2, x_center))
            y_center = max(h_padded/2, min(img_height - h_padded/2, y_center))
            
            boxes.append([x_center, y_center, w_padded, h_padded])
        
        # If no seeds detected, use default box
        if len(boxes) == 0:
            boxes.append([img_width/2, img_height/2, img_width*0.7, img_height*0.7])
        
        return boxes
    
    def process_image(self, image_path, class_name):
        """
        Process a single image and create YOLO annotations
        """
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return None, None
        
        img_height, img_width = img.shape[:2]
        
        # Auto-detect seeds
        boxes = self.auto_detect_seeds(img)
        
        # Convert boxes to YOLO format (normalized)
        yolo_annotations = []
        for box in boxes:
            x_center, y_center, w, h = box
            
            # Normalize to 0-1
            x_norm = x_center / img_width
            y_norm = y_center / img_height
            w_norm = w / img_width
            h_norm = h / img_height
            
            # Clamp values
            x_norm = max(0, min(1, x_norm))
            y_norm = max(0, min(1, y_norm))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))
            
            yolo_annotations.append(f"{self.class_to_id[class_name]} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        return img, yolo_annotations
    
    def create_dataset_split(self, use_augmented=True, train_split=0.8):
        """
        Create train/val split for YOLO training
        """
        # Create directories
        (self.yolo_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.yolo_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (self.yolo_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (self.yolo_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        all_images = []
        
        # Process each class
        for folder_name, yolo_class in self.folder_to_class.items():
            print(f"\nProcessing {folder_name} -> {yolo_class}")
            
            # Original data
            class_dir = self.data_dir / folder_name
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
            
            # Add augmented data if available
            if use_augmented and (self.aug_dir / folder_name).exists():
                aug_files = list((self.aug_dir / folder_name).glob("*.jpg"))
                image_files.extend(aug_files)
                print(f"  Found {len(aug_files)} augmented images")
            
            print(f"  Total images: {len(image_files)}")
            
            # Process each image
            for img_path in tqdm(image_files, desc=f"  Processing"):
                # Generate unique filename
                unique_id = f"{yolo_class}_{img_path.stem}"
                
                # Process image
                img, annotations = self.process_image(img_path, yolo_class)
                if img is None:
                    continue
                
                # Decide train/val split
                is_train = random.random() < train_split
                split = "train" if is_train else "val"
                
                # Save image
                img_filename = f"{unique_id}.jpg"
                img_save_path = self.yolo_dir / "images" / split / img_filename
                cv2.imwrite(str(img_save_path), img)
                
                # Save annotations
                label_filename = f"{unique_id}.txt"
                label_save_path = self.yolo_dir / "labels" / split / label_filename
                
                with open(label_save_path, 'w') as f:
                    for ann in annotations:
                        f.write(ann + "\n")
                
                # Store for dataset info
                all_images.append({
                    'path': str(img_save_path),
                    'split': split,
                    'class': yolo_class,
                    'seed_count': len(annotations)
                })
        
        return all_images
    
    def create_dataset_yaml(self):
        """Create dataset.yaml configuration file for YOLO"""
        config = {
            'path': str(self.yolo_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.classes),
            'names': self.classes
        }
        
        yaml_path = self.yolo_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"\nCreated dataset config: {yaml_path}")
        return config
    
    def visualize_sample_annotations(self, num_samples=3):
        """Visualize annotations for debugging"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            print("Matplotlib not installed. Skipping visualization.")
            return
        
        # Get some sample images
        train_images_dir = self.yolo_dir / "images" / "train"
        sample_images = list(train_images_dir.glob("*.jpg"))[:num_samples]
        
        if not sample_images:
            print("No images found for visualization")
            return
        
        fig, axes = plt.subplots(1, len(sample_images), figsize=(15, 5))
        if len(sample_images) == 1:
            axes = [axes]
        
        for idx, img_path in enumerate(sample_images):
            # Load image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Load annotations
            label_path = self.yolo_dir / "labels" / "train" / f"{img_path.stem}.txt"
            if label_path.exists():
                with open(label_path, 'r') as f:
                    annotations = f.readlines()
                
                # Draw image
                axes[idx].imshow(img)
                axes[idx].set_title(f"{img_path.stem}\n{len(annotations)} seeds")
                
                # Draw bounding boxes
                img_height, img_width = img.shape[:2]
                for ann in annotations:
                    parts = ann.strip().split()
                    if len(parts) == 5:
                        class_id, x_norm, y_norm, w_norm, h_norm = map(float, parts)
                        
                        # Convert normalized to pixel coordinates
                        x_center = x_norm * img_width
                        y_center = y_norm * img_height
                        w = w_norm * img_width
                        h = h_norm * img_height
                        
                        # Calculate top-left corner
                        x = x_center - w/2
                        y = y_center - h/2
                        
                        # Create rectangle
                        rect = patches.Rectangle(
                            (x, y), w, h, 
                            linewidth=2, 
                            edgecolor='red', 
                            facecolor='none'
                        )
                        axes[idx].add_patch(rect)
                        
                        # Add class label
                        axes[idx].text(x, y-5, self.classes[int(class_id)], 
                                     color='red', fontsize=8, backgroundcolor='white')
            
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_annotations.jpg', dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved visualization to: sample_annotations.jpg")
    
    def run(self, use_augmented=True):
        """Run the complete data preparation pipeline"""
        print("=" * 60)
        print("PREPARING YOLO DATASET FOR SESAME SEED DETECTION")
        print("=" * 60)
        
        # Step 1: Create dataset split
        print("\n1. Creating dataset split...")
        images_info = self.create_dataset_split(use_augmented=use_augmented)
        
        # Count statistics
        train_count = sum(1 for img in images_info if img['split'] == 'train')
        val_count = sum(1 for img in images_info if img['split'] == 'val')
        
        print(f"\nDataset Statistics:")
        print(f"  Total images: {len(images_info)}")
        print(f"  Training images: {train_count}")
        print(f"  Validation images: {val_count}")
        
        # Step 2: Create dataset YAML
        print("\n2. Creating dataset configuration...")
        self.create_dataset_yaml()
        
        # Step 3: Visualize samples
        print("\n3. Creating sample visualizations...")
        self.visualize_sample_annotations(num_samples=3)
        
        print("\n" + "=" * 60)
        print("DATA PREPARATION COMPLETE!")
        print("=" * 60)
        print(f"\nYour YOLO dataset is ready at: {self.yolo_dir}")
        print(f"\nNext steps:")
        print("1. Check sample annotations: sample_annotations.jpg")
        print("2. Train YOLO model: python src/train_yolo.py")
        print("3. Dataset config: yolov8/dataset.yaml")


def prepare_yolo_dataset():
    """Main function to run data preparation"""
    preparer = YOLODataPreparer()
    preparer.run(use_augmented=True)


if __name__ == "__main__":
    # Install missing packages
    try:
        import yaml
    except ImportError:
        print("Installing pyyaml...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
    
    prepare_yolo_dataset()