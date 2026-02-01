from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import base64
from pathlib import Path
import io
from detect import SesameSeedDetector
import os
import subprocess
import sys

# Function to ensure model exists
def ensure_model():
    model_path = Path("models/best_yolo.pt")
    
    # Check if model exists AND is not empty
    if not model_path.exists() or model_path.stat().st_size == 0:
        print("\n" + "!"*60)
        print("MODEL NOT FOUND OR CORRUPT! STARTING AUTO-TRAINING...")
        print("!"*60)
        
        # 1. Prepare data
        print("\nStep 1: Preparing dataset...")
        prepare_script = Path("src/prepare_yolo_data.py")
        subprocess.check_call([sys.executable, str(prepare_script)])
        
        # 2. Train model
        print("\nStep 2: Training model (Non-interactive)...")
        from train_yolo import OptimizedSesameTrainer
        trainer = OptimizedSesameTrainer()
        success = trainer.run(interactive=False)
        
        if not success:
            raise RuntimeError("Auto-training failed. Please check logs.")
        
        # 3. Handle model saving/linking
        # The trainer might save to models/best_yolo_optimized.pt
        optimized_model = Path("models/best_yolo_optimized.pt")
        
        # Find the actual best weights if not in the expected destination
        if not optimized_model.exists():
            print("Searching for trained weights...")
            for weight_path in Path("runs").rglob("best.pt"):
                optimized_model = weight_path
                break
        
        if optimized_model.exists():
            import shutil
            # Ensure the empty/corrupt model is replaced
            if model_path.exists():
                model_path.unlink()
            shutil.copy2(optimized_model, model_path)
            print(f"\n✓ Copied model from {optimized_model} to {model_path}")
            
        print("\n✓ Auto-training complete! Model is ready.")

# Initialize app
app = FastAPI(title="Sesame Seed Detection API")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure model exists before starting
ensure_model()

# Function to fix image orientation from EXIF
def fix_orientation(img, contents):
    try:
        from PIL import Image, ExifTags
        image = Image.open(io.BytesIO(contents))
        
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation)
            if orientation == 3:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif orientation == 6:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif orientation == 8:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception as e:
        print(f"EXIF orientation fix failed: {e}")
    return img

# Initialize detector
detector = SesameSeedDetector()

@app.post("/detect")
async def detect_seeds(file: UploadFile = File(...)):
    """
    Upload an image and get seed detection results.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image bytes
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Fix orientation from EXIF
        img = fix_orientation(img, contents)

        # Run detection
        result = detector.detect_image(img, save=False, show=False)

        if result is None:
            raise HTTPException(status_code=500, detail="Detection failed")

        # Encode annotated image to base64
        _, buffer = cv2.imencode('.jpg', result["annotated_image"])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Calculate percentages
        detections = result["detections"]
        total = len(detections)
        stats = {}
        
        if total > 0:
            for name in detector.class_names:
                count = sum(1 for d in detections if d["class"] == name)
                stats[name] = {
                    "count": count,
                    "percentage": round((count / total) * 100, 2)
                }

        return {
            "total_seeds": total,
            "stats": stats,
            "annotated_image": f"data:image/jpeg;base64,{img_base64}",
            "filename": file.filename
        }

    except Exception as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": detector.model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
