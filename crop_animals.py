from ultralytics import YOLO
import os
from glob import glob
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# === Config ===
MODEL_PATH = "yolo_subset/yolov11_subset_results/lr001/weights/best.pt"
INPUT_DIR = "organized_by_species"         # each subfolder is a species
OUTPUT_BASE = "cropped_animals"
CONF_THRESHOLD = 0.5

# === Load YOLO model ===
model = YOLO(MODEL_PATH)

# === Process all species folders ===
for species in os.listdir(INPUT_DIR):
    species_path = os.path.join(INPUT_DIR, species)
    if not os.path.isdir(species_path):
        continue

    output_dir = os.path.join(OUTPUT_BASE, species)
    os.makedirs(output_dir, exist_ok=True)

    image_paths = glob(f"{species_path}/*.jpg")
    for img_path in tqdm(image_paths, desc=f"Cropping {species}"):
        try:
            img = Image.open(img_path).convert("RGB")
        except UnidentifiedImageError:
            print(f"❌ Skipping unreadable image: {img_path}")
            continue

        results = model(img)
        orig_name = os.path.splitext(os.path.basename(img_path))[0]

        for i, box in enumerate(results[0].boxes):
            if box.conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cropped = img.crop((x1, y1, x2, y2))
            crop_path = os.path.join(output_dir, f"{orig_name}_crop{i}.jpg")
            cropped.save(crop_path)
