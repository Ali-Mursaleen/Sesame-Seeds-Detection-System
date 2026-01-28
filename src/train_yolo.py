#!/usr/bin/env python3
"""
Train YOLOv8 model for sesame seed detection
"""

import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO
import yaml
import shutil

class SesameSeedTrainer:
    def __init__(self):
        self.project_dir = Path(".")
        self.data_config = self.project_dir / "yolov8" / "dataset.yaml"
        self.models_dir = self.project_dir / "models"
        self.runs_dir = self.project_dir / "runs"
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        
        # Training configuration
        self.config = {
            'model': 'yolov8n.pt',  # Start with nano model (fastest)
            'epochs': 100,           # Number of training epochs
            'imgsz': 640,            # Input image size
            'batch': 16,             # Batch size
            'workers': 4,            # Data loader workers
            'patience': 30,          # Early stopping patience
            'lr0': 0.01,             # Initial learning rate
            'lrf': 0.01,             # Final learning rate factor
            'momentum': 0.937,       # SGD momentum
            'weight_decay': 0.0005,  # Optimizer weight decay
            'warmup_epochs': 3,      # Warmup epochs
            'warmup_momentum': 0.8,  # Warmup momentum
            'box': 7.5,              # Box loss gain
            'cls': 0.5,              # Class loss gain
            'dfl': 1.5,              # Distribution Focal Loss gain
            'hsv_h': 0.015,          # Image HSV-Hue augmentation
            'hsv_s': 0.7,            # Image HSV-Saturation augmentation
            'hsv_v': 0.4,            # Image HSV-Value augmentation
            'degrees': 0.0,          # Image rotation (+/- deg)
            'translate': 0.1,        # Image translation (+/- fraction)
            'scale': 0.5,            # Image scale (+/- gain)
            'shear': 0.0,            # Image shear (+/- deg)
            'perspective': 0.0,      # Image perspective (+/- fraction)
            'flipud': 0.0,           # Image flip up-down (probability)
            'fliplr': 0.5,           # Image flip left-right (probability)
            'mosaic': 1.0,           # Image mosaic (probability)
            'mixup': 0.0,            # Image mixup (probability)
            'copy_paste': 0.0,       # Segment copy-paste (probability)
            'save': True,            # Save checkpoints
            'save_period': -1,       # Save checkpoint every x epochs
            'cache': False,          # Cache images
            'device': self.get_device(),  # Device selection
            'project': str(self.runs_dir / "detect"),  # Save directory
            'name': 'sesame_train',  # Experiment name
            'exist_ok': True,        # Overwrite existing experiment
            'pretrained': True,      # Use pretrained weights
            'optimizer': 'AdamW',    # Optimizer choice
            'verbose': True,         # Verbose mode
            'seed': 42,              # Random seed
            'single_cls': False,     # Train as single-class dataset
            'rect': False,           # Rectangular training
            'cos_lr': False,         # Cosine LR scheduler
            'close_mosaic': 10,      # Disable mosaic at final epoch
            'resume': False,         # Resume from last checkpoint
            'amp': True,             # Automatic Mixed Precision
            'fraction': 1.0,         # Fraction of dataset to use
            'profile': False,        # Profile ONNX and TensorRT
            'overlap_mask': True,    # Masks should overlap
            'mask_ratio': 4,         # Mask downsample ratio
            'dropout': 0.0,          # Use dropout regularization
            'val': True,             # Validate during training
            'plots': True,           # Save plots during training
        }
    
    def get_device(self):
        """Select the best available device"""
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'  # Apple Silicon
        else:
            return 'cpu'
    
    def check_dataset(self):
        """Check if dataset is properly prepared"""
        print("Checking dataset...")
        
        # Check dataset.yaml exists
        if not self.data_config.exists():
            print(f"❌ Dataset config not found: {self.data_config}")
            print("Please run prepare_yolo_data.py first")
            return False
        
        # Load and check config
        with open(self.data_config, 'r') as f:
            data_cfg = yaml.safe_load(f)
        
        # Check paths
        required_paths = ['train', 'val']
        for path_key in required_paths:
            path = self.project_dir / data_cfg['path'] / data_cfg[path_key]
            if not path.exists():
                print(f"❌ Path not found: {path}")
                return False
            
            # Check if directory has images
            image_files = list(path.glob("*.jpg")) + list(path.glob("*.png")) + list(path.glob("*.jpeg"))
            if len(image_files) == 0:
                print(f"❌ No images found in: {path}")
                return False
            print(f"✓ Found {len(image_files)} images in {path_key}")
        
        # Check labels
        labels_dir = self.project_dir / data_cfg['path'] / "labels"
        if not labels_dir.exists():
            print(f"❌ Labels directory not found: {labels_dir}")
            return False
        
        print("✓ Dataset check passed!")
        return True
    
    def print_training_info(self):
        """Print training configuration information"""
        print("\n" + "="*60)
        print("YOLOv8 TRAINING CONFIGURATION")
        print("="*60)
        
        print(f"\nDataset:")
        print(f"  Config: {self.data_config}")
        
        with open(self.data_config, 'r') as f:
            data_cfg = yaml.safe_load(f)
            print(f"  Classes: {data_cfg['names']}")
            print(f"  Number of classes: {data_cfg['nc']}")
        
        print(f"\nModel:")
        print(f"  Model: {self.config['model']}")
        print(f"  Device: {self.config['device']}")
        print(f"  Image size: {self.config['imgsz']}")
        print(f"  Batch size: {self.config['batch']}")
        print(f"  Epochs: {self.config['epochs']}")
        
        print(f"\nTraining:")
        print(f"  Learning rate: {self.config['lr0']}")
        print(f"  Optimizer: {self.config['optimizer']}")
        print(f"  Augmentation: Enabled")
        print(f"  Early stopping: {self.config['patience']} epochs")
        
        print(f"\nOutput:")
        print(f"  Project: {self.config['project']}")
        print(f"  Name: {self.config['name']}")
        print(f"  Models will be saved to: {self.models_dir}")
        
        print("\n" + "="*60)
    
    def train(self):
        """Train the YOLOv8 model"""
        print("\nStarting training...")
        
        try:
            # Load model
            print(f"Loading model: {self.config['model']}")
            model = YOLO(self.config['model'])
            
            # Train the model
            results = model.train(
                data=str(self.data_config),
                epochs=self.config['epochs'],
                imgsz=self.config['imgsz'],
                batch=self.config['batch'],
                workers=self.config['workers'],
                patience=self.config['patience'],
                lr0=self.config['lr0'],
                lrf=self.config['lrf'],
                momentum=self.config['momentum'],
                weight_decay=self.config['weight_decay'],
                warmup_epochs=self.config['warmup_epochs'],
                warmup_momentum=self.config['warmup_momentum'],
                box=self.config['box'],
                cls=self.config['cls'],
                dfl=self.config['dfl'],
                hsv_h=self.config['hsv_h'],
                hsv_s=self.config['hsv_s'],
                hsv_v=self.config['hsv_v'],
                degrees=self.config['degrees'],
                translate=self.config['translate'],
                scale=self.config['scale'],
                shear=self.config['shear'],
                perspective=self.config['perspective'],
                flipud=self.config['flipud'],
                fliplr=self.config['fliplr'],
                mosaic=self.config['mosaic'],
                mixup=self.config['mixup'],
                copy_paste=self.config['copy_paste'],
                save=self.config['save'],
                save_period=self.config['save_period'],
                cache=self.config['cache'],
                device=self.config['device'],
                project=self.config['project'],
                name=self.config['name'],
                exist_ok=self.config['exist_ok'],
                pretrained=self.config['pretrained'],
                optimizer=self.config['optimizer'],
                verbose=self.config['verbose'],
                seed=self.config['seed'],
                single_cls=self.config['single_cls'],
                rect=self.config['rect'],
                cos_lr=self.config['cos_lr'],
                close_mosaic=self.config['close_mosaic'],
                resume=self.config['resume'],
                amp=self.config['amp'],
                fraction=self.config['fraction'],
                profile=self.config['profile'],
                overlap_mask=self.config['overlap_mask'],
                mask_ratio=self.config['mask_ratio'],
                dropout=self.config['dropout'],
                val=self.config['val'],
                plots=self.config['plots'],
            )
            
            return results
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None
    
    def save_best_model(self):
        """Copy the best model to models directory"""
        # Find the latest training run
        train_runs = list((self.runs_dir / "detect").glob("*"))
        if not train_runs:
            print("No training runs found")
            return
        
        latest_run = max(train_runs, key=os.path.getmtime)
        weights_dir = latest_run / "weights"
        
        if weights_dir.exists():
            # Find best model
            best_model = weights_dir / "best.pt"
            if best_model.exists():
                # Copy to models directory
                dest_path = self.models_dir / "best_yolo.pt"
                shutil.copy2(best_model, dest_path)
                print(f"✓ Best model saved to: {dest_path}")
                
                # Also copy last model
                last_model = weights_dir / "last.pt"
                if last_model.exists():
                    shutil.copy2(last_model, self.models_dir / "last_yolo.pt")
            
            # Copy args.yaml for reference
            args_file = latest_run / "args.yaml"
            if args_file.exists():
                shutil.copy2(args_file, self.models_dir / "training_args.yaml")
    
    def print_training_summary(self):
        """Print summary of training results"""
        # Find the latest training run
        train_runs = list((self.runs_dir / "detect").glob("*"))
        if not train_runs:
            print("No training runs found")
            return
        
        latest_run = max(train_runs, key=os.path.getmtime)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        
        print(f"\nResults saved to: {latest_run}")
        
        # Check for results files
        results_file = latest_run / "results.csv"
        if results_file.exists():
            print("\nTraining metrics available in:")
            print(f"  CSV results: {results_file}")
        
        # Check for plots
        plots = list(latest_run.glob("*.png"))
        if plots:
            print("\nGenerated plots:")
            for plot in plots:
                print(f"  • {plot.name}")
        
        print(f"\nBest model saved to: {self.models_dir}/best_yolo.pt")
        
        print("\nNext steps:")
        print("1. Evaluate model: python src/detect.py --eval")
        print("2. Test on images: python src/detect.py --image test.jpg")
        print("3. View training results in browser: tensorboard --logdir runs")
    
    def run(self):
        """Run complete training pipeline"""
        print("="*60)
        print("SESAME SEED DETECTION - YOLOv8 TRAINING")
        print("="*60)
        
        # Step 1: Check dataset
        if not self.check_dataset():
            print("\n❌ Dataset check failed. Please fix issues above.")
            return False
        
        # Step 2: Print training info
        self.print_training_info()
        
        # Step 3: Start training
        input("\nPress Enter to start training (or Ctrl+C to cancel)...")
        
        try:
            results = self.train()
            if results is None:
                return False
            
            # Step 4: Save best model
            self.save_best_model()
            
            # Step 5: Print summary
            self.print_training_summary()
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            return False


def train_model():
    """Main function to train YOLO model"""
    trainer = SesameSeedTrainer()
    return trainer.run()


if __name__ == "__main__":
    # Check if ultralytics is installed
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    
    # Run training
    success = train_model()
    
    if success:
        print("\n✅ Training completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Training failed!")
        sys.exit(1)