import os
import json
import random
import requests
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

from PytorchWildlife.models import detection as pw_detection

# === CONFIG ===
METADATA_FILE = "wcs_camera_traps.json/wcs_camera_traps.json"
DOWNLOAD_BASE = "https://storage.googleapis.com/public-datasets-lila/wcs-unzipped/"
TEMP_IMAGE_DIR = "temp_images"
CROP_OUTPUT_DIR = "cropped_animals"

NUM_WORKERS = 8
TOP_N_SPECIES = 50
MAX_IMAGES_PER_SPECIES = 20000
MAX_RETRIES = 3

# Create dirs
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
os.makedirs(CROP_OUTPUT_DIR, exist_ok=True)

# === Initialize MegaDetectorV6 ===
print("🔍 Initializing MegaDetectorV6...")
detection_model = pw_detection.MegaDetectorV6(version="MDV6-yolov9-c")
print("✅ MegaDetectorV6 ready.")

# === Helper: Crop and Save ===
def crop_and_save(image, bbox_tuple, output_dir, base_filename):
    try:
        coords = np.array(bbox_tuple[0]).flatten()
        xmin, ymin, xmax, ymax = map(float, coords)
        cropped = image.crop((xmin, ymin, xmax, ymax))
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{base_filename}.jpg")
        cropped.save(out_path)
        return True
    except Exception as e:
        print(f"❌ Crop failed: {e}")
        return False

# === Main ===
def main():
    print("📥 Loading metadata...")
    with open(METADATA_FILE, 'r') as f:
        data = json.load(f)

    # === Map category ID → name
    category_map = {c["id"]: c["name"] for c in data["categories"]}

    # === Identify and exclude 'empty' and 'human' categories
    excluded_ids = set([
        cid for cid, name in category_map.items()
        if name.strip().lower() in {"empty", "human"}
    ])

    # === Build species → image list (excluding empty/human)
    image_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    species_images = defaultdict(list)

    for ann in tqdm(data["annotations"], desc="Indexing"):
        cat_id = ann["category_id"]
        if cat_id in excluded_ids:
            continue
        species = category_map[cat_id]
        species_images[species].append(ann["image_id"])

    # === Select top species
    top_species = sorted(
        [(k, len(v)) for k, v in species_images.items()],
        key=lambda x: -x[1]
    )[:TOP_N_SPECIES]

    species_counts = defaultdict(int)

    print("\n🚀 Processing species...")
    for species, _ in tqdm(top_species, desc="Species"):
        species_dir = os.path.join(CROP_OUTPUT_DIR, species.replace(" ", "_").lower())
        image_ids = random.sample(species_images[species], len(species_images[species]))

        for img_id in image_ids:
            if species_counts[species] >= MAX_IMAGES_PER_SPECIES:
                break

            img_file = image_id_to_file.get(img_id)
            if not img_file:
                continue

            # Flattened filename for uniqueness
            subpath_parts = os.path.normpath(img_file).split(os.sep)
            flat_name = "_".join(subpath_parts)  # e.g., site1_cam1_IMG123.jpg
            temp_path = os.path.join(TEMP_IMAGE_DIR, flat_name)

            # === Download image
            url = f"{DOWNLOAD_BASE}{img_file}"
            success = False
            for _ in range(MAX_RETRIES):
                try:
                    response = requests.get(url, stream=True, timeout=30)
                    response.raise_for_status()
                    with open(temp_path, 'wb') as f:
                        for chunk in response.iter_content(8192):
                            f.write(chunk)
                    success = True
                    break
                except Exception as e:
                    print(f"❌ Download failed: {url} — {e}")
                    continue

            if not success or not os.path.exists(temp_path):
                continue

            # === Run MegaDetector
            try:
                image = Image.open(temp_path).convert("RGB")
                image_array = np.array(image)
                result = detection_model.single_image_detection(image_array)

                detections = result.get("detections", [])
                if len(detections) != 1:
                    os.remove(temp_path)
                    continue  # Skip multi-animal or no-animal images

                crop_success = crop_and_save(
                    image,
                    detections[0],
                    species_dir,
                    flat_name.replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
                )
                if crop_success:
                    species_counts[species] += 1

            except Exception as e:
                print(f"❌ Detection failed: {img_file} — {e}")

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    # === Summary ===
    print("\n✅ Final counts:")
    for species, count in species_counts.items():
        print(f"{species:25s}: {count:5d}/{MAX_IMAGES_PER_SPECIES}")

if __name__ == "__main__":
    main()
