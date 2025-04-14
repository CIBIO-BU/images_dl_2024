from ultralytics import YOLO
import os
from glob import glob
from PIL import Image
from tqdm import tqdm  

# Load model
model = YOLO("yolov11_subset_results/lr001/weights/best.pt")

# Setup paths
base_dir = "images"
output_base = "cropped_animals"

for split in ["train", "val"]:
    output_dir = os.path.join(output_base, split)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process images
    for img_path in tqdm(glob(f"{base_dir}/{split}/*.jpg"), desc=f"Cropping {split}"):
        img = Image.open(img_path)
        results = model(img)
        
        for i, box in enumerate(results[0].boxes):
            if box.conf > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cropped = img.crop((x1, y1, x2, y2))
                
                orig_name = os.path.basename(img_path)[:-4]
                cropped.save(f"{output_dir}/{orig_name}_crop{i}.jpg")