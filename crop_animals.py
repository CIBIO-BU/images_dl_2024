from ultralytics import YOLO
import os
from glob import glob
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Config ===
MODEL_PATH = "yolo_subset/yolov11_subset_results/lr001/weights/best.pt"
INPUT_DIR = "lila_species_organized"         
OUTPUT_BASE = "cropped_animals"
CONF_THRESHOLD = 0.5
BATCH_SIZE = 8  # Adjust based on your GPU memory
NUM_WORKERS = multiprocessing.cpu_count()  # For parallel processing

# === Load YOLO model ===
model = YOLO(MODEL_PATH)

def process_species(species):
    species_path = os.path.join(INPUT_DIR, species)
    output_dir = os.path.join(OUTPUT_BASE, species)
    os.makedirs(output_dir, exist_ok=True)

    image_paths = glob(f"{species_path}/*.jpg")
    
    # Process images in batches
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), 
                 desc=f"Processing {species}", 
                 position=1, leave=False):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        batch_images = []
        valid_paths = []
        
        # Load batch images
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                batch_images.append(img)
                valid_paths.append(img_path)
            except UnidentifiedImageError:
                continue
        
        if not batch_images:
            continue
            
        # Run batch inference
        results = model(batch_images)
        
        # Process results in parallel
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            for img, img_path, result in zip(batch_images, valid_paths, results):
                futures.append(executor.submit(
                    process_result, img, img_path, output_dir, result
                ))
            
            for future in as_completed(futures):
                future.result()  # Just to catch any exceptions

def process_result(img, img_path, output_dir, result):
    orig_name = os.path.splitext(os.path.basename(img_path))[0]
    for i, box in enumerate(result.boxes):
        if box.conf < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cropped = img.crop((x1, y1, x2, y2))
        crop_path = os.path.join(output_dir, f"{orig_name}_crop{i}.jpg")
        cropped.save(crop_path)

# === Main processing ===
if __name__ == "__main__":
    species_list = [d for d in os.listdir(INPUT_DIR) 
                   if os.path.isdir(os.path.join(INPUT_DIR, d))]
    
    print(f"Starting processing of {len(species_list)} species...")
    with multiprocessing.Pool(processes=min(4, NUM_WORKERS)) as pool:  # Limit to 4 processes to avoid memory issues
        list(tqdm(pool.imap(process_species, species_list), 
                 total=len(species_list), 
                 desc="Overall Progress"))
    print("Processing complete!")