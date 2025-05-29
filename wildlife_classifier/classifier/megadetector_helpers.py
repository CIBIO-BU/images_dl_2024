import json
import subprocess

def run_megadetector(image_path, detection_threshold=0.5):
    """
    Call MegaDetector CLI (or a custom wrapper) and parse output.
    You must have the MegaDetector CLI or Python runner installed and configured.
    """
    results_path = image_path + "_detections.json"

    # Run CLI tool or use a Python model
    command = [
        "python", "run_detector.py",  # Adjust this path if needed
        "md_v5a.0.0.pt",              # Your MegaDetector model file
        image_path,
        results_path,
        "--threshold", str(detection_threshold),
        "--output_relative_filenames"
    ]
    subprocess.run(command, check=True)

    with open(results_path) as f:
        results = json.load(f)

    detections = []
    for det in results["images"][0]["detections"]:
        if det["conf"] < detection_threshold:
            continue
        bbox = det["bbox"]  # Format: [x, y, width, height] normalized
        width, height = results["images"][0]["width"], results["images"][0]["height"]
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int((bbox[0] + bbox[2]) * width)
        y2 = int((bbox[1] + bbox[3]) * height)
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "conf": det["conf"]
        })

    return detections
