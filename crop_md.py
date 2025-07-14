import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from PytorchWildlife.models import detection as pw_detection

# Initialize model
print("Initializing MegaDetectorV6...")
detection_model = pw_detection.MegaDetectorV6(version="MDV6-yolov9-c")
print("Model initialized.")

INPUT_FOLDER = "snapshot_safari_10k"
OUTPUT_FOLDER = "snapshot_safari_10k_crops"

def crop_and_save(image, bbox_tuple, label, output_dir, base_filename, index):
    try:
        # bbox_tuple is like (array([xmin, ymin, xmax, ymax]), None, conf, label_id, None, {})
        bbox_coords = bbox_tuple[0]  # Extract the array from the first element

        # Convert bbox_coords to floats for cropping
        xmin, ymin, xmax, ymax = map(float, bbox_coords)

        print(f"  Cropping detection {index}: bbox = ({xmin}, {ymin}, {xmax}, {ymax}), label = {label}")

        cropped = image.crop((xmin, ymin, xmax, ymax))

        safe_label = str(label).replace("/", "_").replace("\\", "_").replace(" ", "_")
        output_path = os.path.join(output_dir, f"{base_filename}_det{index}_{safe_label}.jpg")
        cropped.save(output_path)
        print(f"  ✅ Saved cropped image: {output_path}")

    except Exception as e:
        print(f"  ❌ Error cropping/saving detection {index}: {e}")

def process_folder(input_dir, output_dir):
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(root, f)
                image_paths.append(full_path)

    print(f"Found {len(image_paths)} image files in '{input_dir}' (including subfolders)")

    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            rel_path = os.path.relpath(image_path, input_dir)
            subfolder = os.path.dirname(rel_path)
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image)

            print(f"\n🔍 Processing: {rel_path}")
            result = detection_model.single_image_detection(image_array)

            print(f"  Result keys: {result.keys()}")
            print(f"  # Detections: {len(result.get('detections', []))}")
            print(f"  # Labels: {len(result.get('labels', []))}")

            if not result.get("detections"):
                print("  ⚠️ No detections found.")
                continue

            # Create corresponding output subfolder
            output_subfolder = os.path.join(output_dir, subfolder)
            os.makedirs(output_subfolder, exist_ok=True)

            base_filename = os.path.splitext(os.path.basename(image_path))[0]

            for i, (bbox, label) in enumerate(zip(result["detections"], result["labels"])):
                crop_and_save(image, bbox, label, output_subfolder, base_filename, i)

        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")

# Run it
if __name__ == "__main__":
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER)
