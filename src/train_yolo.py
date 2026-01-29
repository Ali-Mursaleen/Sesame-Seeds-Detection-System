#!/usr/bin/env python3
"""
OPTIMIZED YOLOv8 training for low-spec PCs
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import yaml
import torch

class OptimizedSesameTrainer:
    def __init__(self):
        self.project_dir = Path(".")
        self.data_config = self.project_dir / "yolov8" / "dataset.yaml"
        self.models_dir = self.project_dir / "models"
        self.runs_dir = self.project_dir / "runs"
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        
        # OPTIMIZED CONFIG FOR LOW-SPEC PCS
        self.config = {
            'model': 'yolov8n.pt',  # NANO model (smallest)
            'epochs': 30,            # REDUCED epochs (from 100)
            'imgsz': 320,           # REDUCED image size (from 640)
            'batch': 4,             # REDUCED batch size (from 16)
            'workers': 2,           # REDUCED workers (from 4)
            'patience': 10,         # Early stopping patience
            'device': self.get_device(),
            'project': str(self.runs_dir / "detect"),
            'name': 'sesame_optimized',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'SGD',     # SGD uses less memory than AdamW
            'lr0': 0.01,           # Learning rate
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'close_mosaic': 10,
            'amp': False,          # DISABLE mixed precision (causes issues on CPU)
            'cache': False,        # DISABLE caching (saves RAM)
            'verbose': False,      # Less verbose output
            'save': True,
            'save_period': -1,
            'val': True,
            'plots': False,        # DISABLE plots during training (saves RAM)
            'rect': False,
            'cos_lr': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'resume': False,
        }
        
        # Print system info
        self.print_system_info()
    
    def print_system_info(self):
        """Print system information for debugging"""
        print("="*60)
        print("SYSTEM INFORMATION")
        print("="*60)
        print(f"Python: {sys.version}")
        print(f"PyTorch: {torch.__version__}")
        print(f"Device: {self.get_device()}")
        
        if torch.cuda.is_available():
            print(f"CUDA: Available")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("CUDA: Not available (using CPU)")
        
        # Check RAM
        import psutil
        ram = psutil.virtual_memory()
        print(f"RAM: {ram.total / 1e9:.2f} GB total, {ram.available / 1e9:.2f} GB available")
        print("="*60)
    
    def get_device(self):
        """Select device, prefer CPU for stability"""
        # Force CPU for low-spec PCs
        # Remove the comment below if you have a good GPU
        # if torch.cuda.is_available():
        #     return 'cuda'
        return 'cpu'  # Force CPU training
    
    def check_dataset(self):
        """Check if dataset is properly prepared"""
        print("\nChecking dataset...")
        
        if not self.data_config.exists():
            print(f"❌ Dataset config not found: {self.data_config}")
            return False
        
        # Load config
        with open(self.data_config, 'r') as f:
            data_cfg = yaml.safe_load(f)
        
        # Check paths
        train_path = self.project_dir / data_cfg['path'] / data_cfg['train']
        val_path = self.project_dir / data_cfg['path'] / data_cfg['val']
        
        if not train_path.exists() or not val_path.exists():
            print(f"❌ Train/val paths not found")
            return False
        
        # Count images
        train_images = len(list(train_path.glob("*.jpg")))
        val_images = len(list(val_path.glob("*.jpg")))
        
        print(f"✓ Training images: {train_images}")
        print(f"✓ Validation images: {val_images}")
        
        if train_images == 0:
            print("❌ No training images found")
            return False
        
        return True
    
    def reduce_dataset_size(self, keep_fraction=0.5):
        """
        Reduce dataset size for faster training on low-spec PCs
        Only keep a fraction of the data
        """
        print(f"\nReducing dataset size (keeping {keep_fraction*100}%)...")
        
        with open(self.data_config, 'r') as f:
            data_cfg = yaml.safe_load(f)
        
        train_dir = self.project_dir / data_cfg['path'] / data_cfg['train']
        val_dir = self.project_dir / data_cfg['path'] / data_cfg['val']
        
        # Reduce training images
        train_images = list(train_dir.glob("*.jpg"))
        keep_count = int(len(train_images) * keep_fraction)
        
        import random
        random.shuffle(train_images)
        images_to_remove = train_images[keep_count:]
        
        print(f"Keeping {keep_count} of {len(train_images)} training images")
        
        # Remove excess images and their labels
        for img_path in images_to_remove:
            # Remove image
            if img_path.exists():
                img_path.unlink()
            
            # Remove corresponding label
            label_path = (self.project_dir / data_cfg['path'] / 'labels' / 'train' / 
                         f"{img_path.stem}.txt")
            if label_path.exists():
                label_path.unlink()
        
        return True
    
    def train_lightweight(self):
        """Ultra-lightweight training for very low-spec PCs"""
        print("\n" + "="*60)
        print("ULTRA-LIGHTWEIGHT TRAINING MODE")
        print("="*60)
        
        # Further reduce settings
        self.config.update({
            'epochs': 20,
            'imgsz': 224,      # Even smaller
            'batch': 2,        # Tiny batch
            'workers': 1,      # Single worker
            'patience': 5,
        })
        
        # Reduce dataset
        self.reduce_dataset_size(keep_fraction=0.3)
        
        return self.train()
    
    def train(self):
        """Train the model with optimized settings"""
        print("\n" + "="*60)
        print("STARTING OPTIMIZED TRAINING")
        print("="*60)
        print(f"Model: {self.config['model']}")
        print(f"Device: {self.config['device']}")
        print(f"Image size: {self.config['imgsz']}")
        print(f"Batch size: {self.config['batch']}")
        print(f"Epochs: {self.config['epochs']}")
        print("="*60)
        
        try:
            # Load model
            print("Loading model...")
            model = YOLO(self.config['model'])
            
            # Monitor memory during training
            import threading
            stop_monitoring = False
            
            def monitor_memory():
                import psutil
                import time
                while not stop_monitoring:
                    ram = psutil.virtual_memory()
                    print(f"RAM Usage: {ram.used / 1e9:.2f} GB / {ram.total / 1e9:.2f} GB ({ram.percent}%)")
                    time.sleep(30)  # Check every 30 seconds
            
            # Start memory monitoring thread
            monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
            monitor_thread.start()
            
            # Train with minimal settings
            print("\nTraining started... (This may take 15-30 minutes)")
            print("Press Ctrl+C to stop early if needed")
            
            results = model.train(
                data=str(self.data_config),
                epochs=self.config['epochs'],
                imgsz=self.config['imgsz'],
                batch=self.config['batch'],
                workers=self.config['workers'],
                patience=self.config['patience'],
                device=self.config['device'],
                project=self.config['project'],
                name=self.config['name'],
                exist_ok=self.config['exist_ok'],
                pretrained=self.config['pretrained'],
                optimizer=self.config['optimizer'],
                lr0=self.config['lr0'],
                momentum=self.config['momentum'],
                weight_decay=self.config['weight_decay'],
                warmup_epochs=self.config['warmup_epochs'],
                box=self.config['box'],
                cls=self.config['cls'],
                dfl=self.config['dfl'],
                close_mosaic=self.config['close_mosaic'],
                amp=self.config['amp'],
                cache=self.config['cache'],
                verbose=self.config['verbose'],
                save=self.config['save'],
                save_period=self.config['save_period'],
                val=self.config['val'],
                plots=self.config['plots'],
                rect=self.config['rect'],
                cos_lr=self.config['cos_lr'],
                overlap_mask=self.config['overlap_mask'],
                mask_ratio=self.config['mask_ratio'],
                dropout=self.config['dropout'],
                resume=self.config['resume'],
            )
            
            stop_monitoring = True
            monitor_thread.join(timeout=5)
            
            return results
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Training interrupted by user")
            print("Saving current model...")
            stop_monitoring = True
            return None
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            return None
    
    def save_best_model(self):
        """Copy the best model"""
        train_runs = list((self.runs_dir / "detect").glob("*"))
        if not train_runs:
            return
        
        latest_run = max(train_runs, key=lambda x: x.stat().st_mtime)
        weights_dir = latest_run / "weights"
        
        if weights_dir.exists():
            best_model = weights_dir / "best.pt"
            if best_model.exists():
                import shutil
                dest_path = self.models_dir / "best_yolo_optimized.pt"
                shutil.copy2(best_model, dest_path)
                print(f"\n✅ Best model saved to: {dest_path}")
                
                # Also copy config
                args_file = latest_run / "args.yaml"
                if args_file.exists():
                    shutil.copy2(args_file, self.models_dir / "training_args.yaml")
    
    def run(self, mode='optimized'):
        """Run training pipeline"""
        print("\n" + "="*60)
        print("SESAME SEED DETECTION - OPTIMIZED TRAINING")
        print("="*60)
        
        # Check dataset
        if not self.check_dataset():
            print("\n❌ Dataset check failed")
            return False
        
        # Ask for mode
        print("\nSelect training mode:")
        print("1. Optimized (Recommended for most PCs)")
        print("2. Ultra-lightweight (For very low-spec PCs)")
        print("3. Custom settings")
        
        choice = input("\nEnter choice (1-3, default=1): ").strip()
        
        if choice == '2':
            results = self.train_lightweight()
        elif choice == '3':
            self.custom_settings()
            results = self.train()
        else:
            results = self.train()
        
        if results is not None:
            self.save_best_model()
            self.print_summary()
            return True
        
        return False
    
    def custom_settings(self):
        """Allow user to set custom training parameters"""
        print("\n" + "="*60)
        print("CUSTOM TRAINING SETTINGS")
        print("="*60)
        
        print(f"Current image size: {self.config['imgsz']}")
        new_size = input("Enter image size (224, 320, 416, 512, 640): ").strip()
        if new_size.isdigit():
            self.config['imgsz'] = int(new_size)
        
        print(f"\nCurrent batch size: {self.config['batch']}")
        new_batch = input("Enter batch size (1-16, lower=less memory): ").strip()
        if new_batch.isdigit():
            self.config['batch'] = int(new_batch)
        
        print(f"\nCurrent epochs: {self.config['epochs']}")
        new_epochs = input("Enter number of epochs (10-100): ").strip()
        if new_epochs.isdigit():
            self.config['epochs'] = int(new_epochs)
        
        print("\nUpdated settings:")
        print(f"Image size: {self.config['imgsz']}")
        print(f"Batch size: {self.config['batch']}")
        print(f"Epochs: {self.config['epochs']}")
    
    def print_summary(self):
        """Print training summary"""
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print("\n✅ Model trained successfully!")
        print(f"📁 Model saved to: models/best_yolo_optimized.pt")
        print(f"📊 Training logs: runs/detect/sesame_optimized/")
        print("\n🚀 Next: Test your model:")
        print("python src/detect.py --image test.jpg --model models/best_yolo_optimized.pt")


def train_model():
    """Main function"""
    trainer = OptimizedSesameTrainer()
    return trainer.run()


if __name__ == "__main__":
    # Install psutil for memory monitoring
    try:
        import psutil
    except ImportError:
        print("Installing psutil for memory monitoring...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    
    success = train_model()
    
    if success:
        print("\n🎉 Optimization complete! Your PC should handle this well.")
        sys.exit(0)
    else:
        print("\n❌ Training failed")
        sys.exit(1)