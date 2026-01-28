#!/usr/bin/env python3
"""
Detect sesame seeds using trained YOLOv8 model
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import sys
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class SesameSeedDetector:
    def __init__(self, model_path=None):
        """
        Initialize detector with trained model
        """
        self.project_dir = Path(".")
        
        # Default model path
        if model_path is None:
            self.model_path = self.project_dir / "models" / "best_yolo.pt"
            if not self.model_path.exists():
                # Try to find model in runs directory
                runs_dir = self.project_dir / "runs" / "detect"
                if runs_dir.exists():
                    train_folders = list(runs_dir.glob("*"))
                    if train_folders:
                        latest_run = max(train_folders, key=lambda x: x.stat().st_mtime)
                        model_path = latest_run / "weights" / "best.pt"
                        if model_path.exists():
                            self.model_path = model_path
        
        # Load model
        print(f"Loading model from: {self.model_path}")
        try:
            self.model = YOLO(str(self.model_path))
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("Please train the model first: python src/train_yolo.py")
            sys.exit(1)
        
        # Class names
        self.class_names = ['healthy', 'black', 'rain_damaged']
        self.class_colors = {
            'healthy': (0, 255, 0),      # Green
            'black': (0, 0, 255),        # Red
            'rain_damaged': (255, 165, 0) # Orange
        }
        
        # Detection confidence threshold
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
    
    def detect_image(self, image_path, save=True, show=True):
        """
        Detect sesame seeds in a single image
        """
        print(f"\nProcessing: {image_path}")
        
        # Read image
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"❌ Could not read image: {image_path}")
                return None
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Assume it's already a numpy array
            image = image_path
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run detection
        results = self.model.predict(
            source=image_rgb,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=640,
            verbose=False
        )
        
        # Process results
        detections = []
        annotated_image = image.copy()
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                boxes = result.boxes.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    
                    # Get class name
                    class_name = self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}"
                    
                    # Store detection
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class': class_name,
                        'class_id': cls,
                        'area': (x2 - x1) * (y2 - y1)
                    })
                    
                    # Draw bounding box
                    color = self.class_colors.get(class_name, (255, 255, 255))
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"{class_name}: {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(annotated_image, 
                                (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), 
                                color, -1)
                    cv2.putText(annotated_image, label, 
                              (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add summary text
        summary = self._get_summary_text(detections)
        self._add_summary_to_image(annotated_image, summary)
        
        # Save annotated image
        if save and isinstance(image_path, (str, Path)):
            output_path = Path(image_path).parent / f"{Path(image_path).stem}_detected.jpg"
            cv2.imwrite(str(output_path), annotated_image)
            print(f"✓ Saved annotated image: {output_path}")
        
        # Show image
        if show:
            plt.figure(figsize=(12, 8))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            plt.title("Detected Seeds")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return {
            'detections': detections,
            'summary': summary,
            'annotated_image': annotated_image,
            'total_seeds': len(detections)
        }
    
    def detect_batch(self, input_dir, output_dir=None):
        """
        Detect seeds in all images in a directory
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"❌ Input directory not found: {input_dir}")
            return
        
        # Create output directory
        if output_dir is None:
            output_dir = input_path.parent / f"{input_path.name}_detected"
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(ext))
        
        if not image_files:
            print(f"❌ No images found in: {input_dir}")
            return
        
        print(f"Found {len(image_files)} images")
        
        # Process each image
        all_results = []
        for img_file in image_files:
            result = self.detect_image(
                img_file, 
                save=True, 
                show=False
            )
            
            if result:
                result['filename'] = img_file.name
                all_results.append(result)
                
                # Save to output directory
                output_file = output_path / f"{img_file.stem}_detected.jpg"
                cv2.imwrite(str(output_file), result['annotated_image'])
        
        # Generate summary report
        self._generate_batch_report(all_results, output_path)
        
        return all_results
    
    def detect_video(self, video_path, output_path=None, show=True):
        """
        Detect seeds in a video file
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect seeds in frame
            result = self.detect_image(frame, save=False, show=False)
            
            if output_path:
                out.write(result['annotated_image'])
            
            if show and frame_count % 30 == 0:  # Show every 30th frame
                cv2.imshow('Detection', result['annotated_image'])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"✓ Processed {frame_count} frames")
    
    def detect_webcam(self, camera_id=0):
        """
        Real-time detection from webcam
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Could not open camera {camera_id}")
            return
        
        print("Starting webcam detection...")
        print("Press 'q' to quit, 's' to save frame")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect seeds
            result = self.detect_image(frame, save=False, show=False)
            
            # Show frame
            cv2.imshow('Sesame Seed Detection', result['annotated_image'])
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"webcam_frame_{frame_count}.jpg"
                cv2.imwrite(filename, result['annotated_image'])
                print(f"Saved frame to {filename}")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam detection stopped")
    
    def _get_summary_text(self, detections):
        """Generate summary text from detections"""
        if not detections:
            return "No seeds detected"
        
        # Count by class
        class_counts = {}
        total_confidence = 0
        
        for det in detections:
            cls = det['class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
            total_confidence += det['confidence']
        
        # Build summary
        summary_lines = [f"Total seeds: {len(detections)}"]
        for cls, count in sorted(class_counts.items()):
            summary_lines.append(f"{cls.capitalize()}: {count}")
        
        if detections:
            avg_confidence = total_confidence / len(detections)
            summary_lines.append(f"Avg confidence: {avg_confidence:.2f}")
        
        return "\n".join(summary_lines)
    
    def _add_summary_to_image(self, image, summary_text):
        """Add summary text to image"""
        lines = summary_text.split('\n')
        y0 = 30
        dy = 25
        
        for i, line in enumerate(lines):
            y = y0 + i * dy
            cv2.putText(image, line, (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)  # Black outline
            cv2.putText(image, line, (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # White text
    
    def _generate_batch_report(self, results, output_dir):
        """Generate CSV report for batch processing"""
        if not results:
            return
        
        report_data = []
        
        for result in results:
            filename = result.get('filename', 'unknown')
            total_seeds = result['total_seeds']
            
            # Count by class
            class_counts = {}
            for det in result['detections']:
                cls = det['class']
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            # Add to report
            row = {'filename': filename, 'total_seeds': total_seeds}
            for cls in self.class_names:
                row[f'{cls}_count'] = class_counts.get(cls, 0)
            
            report_data.append(row)
        
        # Create DataFrame and save CSV
        df = pd.DataFrame(report_data)
        csv_path = output_dir / "detection_report.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"✓ Generated report: {csv_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"Total images processed: {len(results)}")
        print(f"Total seeds detected: {df['total_seeds'].sum()}")
        
        for cls in self.class_names:
            cls_total = df[f'{cls}_count'].sum()
            print(f"Total {cls} seeds: {cls_total}")
        
        print(f"Detailed report saved to: {csv_path}")
    
    def evaluate_model(self, data_dir="yolov8/images/val"):
        """
        Evaluate model on validation/test set
        """
        print("\nEvaluating model...")
        
        val_dir = Path(data_dir)
        if not val_dir.exists():
            print(f"❌ Validation directory not found: {data_dir}")
            return
        
        # Get validation images
        image_files = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.jpeg")) + list(val_dir.glob("*.png"))
        
        if not image_files:
            print(f"❌ No validation images found in: {data_dir}")
            return
        
        print(f"Found {len(image_files)} validation images")
        
        # Run evaluation
        metrics = self.model.val(
            data='yolov8/dataset.yaml',
            split='val',
            imgsz=640,
            batch=16,
            conf=0.25,
            iou=0.45,
            device=self.model.device,
            verbose=True
        )
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        if hasattr(metrics, 'box'):
            print(f"mAP50: {metrics.box.map50:.4f}")
            print(f"mAP50-95: {metrics.box.map:.4f}")
            print(f"Precision: {metrics.box.mp:.4f}")
            print(f"Recall: {metrics.box.mr:.4f}")
        
        return metrics


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Detect sesame seeds using YOLOv8')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--dir', type=str, help='Directory of images to process')
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for real-time detection')
    parser.add_argument('--eval', action='store_true', help='Evaluate model on validation set')
    parser.add_argument('--model', type=str, default=None, help='Path to custom model')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--output', type=str, help='Output directory/file')
    parser.add_argument('--no-show', action='store_true', help='Do not show images')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = SesameSeedDetector(model_path=args.model)
    
    if args.conf:
        detector.conf_threshold = args.conf
    
    if args.eval:
        # Evaluate model
        detector.evaluate_model()
    
    elif args.image:
        # Single image detection
        result = detector.detect_image(
            args.image, 
            save=True, 
            show=not args.no_show
        )
        
        if result:
            print("\n" + "="*60)
            print("DETECTION RESULTS")
            print("="*60)
            print(result['summary'])
            print(f"Total seeds: {result['total_seeds']}")
    
    elif args.dir:
        # Batch processing
        detector.detect_batch(args.dir, args.output)
    
    elif args.video:
        # Video processing
        detector.detect_video(
            args.video, 
            output_path=args.output,
            show=not args.no_show
        )
    
    elif args.webcam:
        # Webcam detection
        detector.detect_webcam()
    
    else:
        # Show help if no arguments
        parser.print_help()
        print("\nExamples:")
        print("  Detect single image: python src/detect.py --image test.jpg")
        print("  Process folder: python src/detect.py --dir images/")
        print("  Use webcam: python src/detect.py --webcam")
        print("  Evaluate model: python src/detect.py --eval")


if __name__ == "__main__":
    # Check if ultralytics is installed
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    
    main()