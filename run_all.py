#!/usr/bin/env python3
"""
Master script to run the entire Sesame Seed Detection pipeline
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

def install_requirements():
    """Install all required packages"""
    print("\n" + "="*60)
    print("INSTALLING REQUIREMENTS")
    print("="*60)
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    print("Installing packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_environment():
    """Check if all required packages are installed"""
    print("\n" + "="*60)
    print("CHECKING ENVIRONMENT")
    print("="*60)
    
    required_packages = [
        ('ultralytics', 'YOLOv8'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('tqdm', 'Progress bars'),
        ('albumentations', 'Augmentations'),
        ('matplotlib', 'Visualization'),
        ('PIL', 'Pillow'),
        ('pandas', 'Data analysis'),
        ('yaml', 'YAML config')
    ]
    
    all_ok = True
    for package, description in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'yaml':
                import yaml
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
            print(f"✅ {package} ({description})")
        except ImportError:
            print(f"❌ {package} ({description}) - MISSING")
            all_ok = False
    
    return all_ok

def run_data_preparation():
    """Run data preparation script"""
    print("\n" + "="*60)
    print("PREPARING DATA")
    print("="*60)
    
    script_path = Path("src/prepare_yolo_data.py")
    if not script_path.exists():
        print("❌ Data preparation script not found!")
        return False
    
    try:
        subprocess.check_call([sys.executable, str(script_path)])
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Data preparation failed: {e}")
        return False

def run_training():
    """Run training script"""
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    script_path = Path("src/train_yolo.py")
    if not script_path.exists():
        print("❌ Training script not found!")
        return False
    
    try:
        subprocess.check_call([sys.executable, str(script_path)])
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return False

def run_detection_test():
    """Run a quick detection test"""
    print("\n" + "="*60)
    print("TESTING DETECTION")
    print("="*60)
    
    script_path = Path("src/detect.py")
    if not script_path.exists():
        print("❌ Detection script not found!")
        return False
    
    # Find a test image
    test_images = list(Path("data").glob("**/*.jpeg")) + list(Path("data").glob("**/*.jpg"))
    
    if test_images:
        test_image = test_images[0]
        print(f"Testing detection on: {test_image}")
        
        try:
            subprocess.check_call([
                sys.executable, str(script_path), 
                "--image", str(test_image),
                "--no-show"
            ])
            
            # Check if output was created
            output_image = Path(f"{test_image.stem}_detected.jpg")
            if output_image.exists():
                print(f"✅ Detection successful! Output saved to: {output_image}")
                return True
            else:
                print("❌ Detection output not found")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Detection test failed: {e}")
            return False
    else:
        print("❌ No test images found in data/ directory")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run complete Sesame Seed Detection pipeline')
    parser.add_argument('--install', action='store_true', help='Install requirements only')
    parser.add_argument('--check', action='store_true', help='Check environment only')
    parser.add_argument('--prepare', action='store_true', help='Prepare data only')
    parser.add_argument('--train', action='store_true', help='Train model only')
    parser.add_argument('--test', action='store_true', help='Test detection only')
    parser.add_argument('--skip-install', action='store_true', help='Skip installation check')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("SESAME SEED DETECTION SYSTEM - COMPLETE PIPELINE")
    print("="*60)
    
    # If specific mode selected, run only that
    if args.install:
        install_requirements()
        return
    
    if args.check:
        check_environment()
        return
    
    if args.prepare:
        run_data_preparation()
        return
    
    if args.train:
        run_training()
        return
    
    if args.test:
        run_detection_test()
        return
    
    # Otherwise run complete pipeline
    print("\nRunning complete pipeline...")
    
    # Step 1: Install requirements (optional)
    if not args.skip_install:
        if not check_environment():
            print("\nSome packages are missing. Installing requirements...")
            install_requirements()
        else:
            print("\n✅ All required packages are installed!")
    
    # Step 2: Prepare data
    print("\n" + "="*60)
    print("STEP 1: DATA PREPARATION")
    print("="*60)
    
    if not run_data_preparation():
        print("\n❌ Data preparation failed. Exiting.")
        return
    
    # Step 3: Train model
    print("\n" + "="*60)
    print("STEP 2: MODEL TRAINING")
    print("="*60)
    
    if not run_training():
        print("\n❌ Training failed. Exiting.")
        return
    
    # Step 4: Test detection
    print("\n" + "="*60)
    print("STEP 3: DETECTION TEST")
    print("="*60)
    
    if not run_detection_test():
        print("\n❌ Detection test failed.")
        return
    
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\n📁 Your project structure:")
    print("├── data/                    # Original images")
    print("├── yolov8/                  # YOLO formatted data")
    print("├── models/                  # Trained models")
    print("│   └── best_yolo.pt        # Your trained model")
    print("├── runs/                    # Training results")
    print("└── src/                     # Source code")
    
    print("\n🚀 Next steps:")
    print("1. Test on more images: python src/detect.py --image your_image.jpg")
    print("2. Process a folder: python src/detect.py --dir images_folder/")
    print("3. Use webcam: python src/detect.py --webcam")
    print("4. Evaluate model: python src/detect.py --eval")

if __name__ == "__main__":
    main()