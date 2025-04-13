from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import tempfile
import os
from django.conf import settings
import base64
from io import BytesIO

# Load the Faster R-CNN model (pre-trained on COCO)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

# COCO class labels (81 classes, including background)
COCO_LABELS = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Ensure the "cropped" folder exists
CROPPED_IMAGES_DIR = os.path.join(settings.BASE_DIR, "cropped")
os.makedirs(CROPPED_IMAGES_DIR, exist_ok=True)

@csrf_exempt
def predict(request):
    if request.method == "POST" and request.FILES.get("file"):
        image = request.FILES["file"]

        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            for chunk in image.chunks():
                temp_file.write(chunk)
            temp_file_path = temp_file.name

        # Load and preprocess the image
        image_data = Image.open(temp_file_path).convert("RGB")
        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(image_data)

        # Run detection on the image
        with torch.no_grad():
            predictions = model([image_tensor])

        # Filter detections above a confidence threshold (e.g., 0.5) and only for animals
        detections_above_threshold = []
        for box, label, score in zip(predictions[0]["boxes"], predictions[0]["labels"], predictions[0]["scores"]):
            if score > 0.5 and COCO_LABELS[label.item()] in ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "person"]:
                # Convert bounding box coordinates to integers
                x_min, y_min, x_max, y_max = map(int, box.tolist())

                # Crop the image using the bounding box
                cropped_image = image_data.crop((x_min, y_min, x_max, y_max))

                print(f"Detected {COCO_LABELS[label.item()]} with confidence {score:.2f}")
                # Save the cropped image to the "cropped" folder
                cropped_image_filename = f"cropped_{len(detections_above_threshold)}_{os.path.basename(temp_file_path)}"
                cropped_image_path = os.path.join(CROPPED_IMAGES_DIR, cropped_image_filename)
                cropped_image.save(cropped_image_path, format="JPEG")

                detections_above_threshold.append({
                    "bbox": [x_min, y_min, x_max, y_max],  # Bounding box coordinates
                    "confidence": float(score),  # Confidence score
                    "label": COCO_LABELS[label.item()],  # Class label
                    "cropped_image_path": cropped_image_path  # Path to the cropped image
                })

        # Clean up the temporary uploaded file
        os.unlink(temp_file_path)

        return JsonResponse({"detections": detections_above_threshold})
    return JsonResponse({"error": "Invalid request"}, status=400)