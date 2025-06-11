from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.conf import settings
import torch
import torchvision
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import tempfile
import numpy as np
import os
import json
from .models import Feedback
from ultralytics import YOLO
from PytorchWildlife.models import detection as pw_detection

# Load models
yolo_model = YOLO(os.path.join(settings.BASE_DIR, 'classifier', 'models', 'crop.pt'))
yolo_model.model.eval()

md_model = pw_detection.MegaDetectorV6(version="MDV6-yolov9-c")

resnet = models.resnet50(weights=None)
resnet.fc = torch.nn.Sequential(
    torch.nn.Linear(resnet.fc.in_features, 1024),
    torch.nn.BatchNorm1d(1024),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(1024, 15)
)
checkpoint_path = os.path.join(settings.BASE_DIR, 'classifier', 'models', '15_s.pth')
checkpoint = torch.load(checkpoint_path, map_location='cpu')
resnet.load_state_dict(checkpoint['model_state_dict'])
resnet.eval()

CLASS_NAMES = [
    'aepyceros melampus', 'bos taurus', 'cephalophus nigrifrons', 'crax rubra', 'dasyprocta punctata',
    'didelphis pernigra', 'equus quagga', 'leopardus pardalis', 'loxodonta africana', 'madoqua guentheri',
    'meleagris ocellata', 'mitu tuberosum', 'panthera onca', 'pecari tajacu', 'tayassu pecari'
]

classification_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@csrf_exempt
def predict(request):
    if request.method == "POST" and request.FILES.get("file"):
        image = request.FILES["file"]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            for chunk in image.chunks():
                temp_file.write(chunk)
            temp_file_path = temp_file.name

        original_image = Image.open(temp_file_path).convert("RGB")
        image_array = np.array((Image.open(temp_file_path)).convert("RGB"))
        results = yolo_model.predict(source=temp_file_path, save=False, imgsz=640, conf=0.5)

        md_results = md_model.single_image_detection(image_array)
        detections = results[0].boxes.xyxy
        confidences = results[0].boxes.conf
        class_ids = results[0].boxes.cls

        detection_results = []

        for bbox, label in zip(md_results['detections'], md_results['labels']):
            x1, y1, x2, y2 = map(float,bbox[0])
            print(f"MD Detection: {label} at [{x1}, {y1}, {x2}, {y2}]")
            cropped_img = original_image.crop((x1, y1, x2, y2))
            input_tensor = classification_transform(cropped_img).unsqueeze(0)

            with torch.no_grad():
                output = resnet(input_tensor)
                predicted_idx = torch.argmax(output, dim=1).item()
                predicted_prob = torch.softmax(output, dim=1)[0, predicted_idx].item()

            detection_info = {
                "bbox": [x1, y1, x2, y2],
                "species_prediction": CLASS_NAMES[predicted_idx],
                "species_confidence": round(float(predicted_prob), 4),
            }
            detection_results.append(detection_info)

        os.unlink(temp_file_path)

        return JsonResponse({
            "detections": detection_results,
            "num_detections": len(detection_results)
        })

    return JsonResponse({"error": "Invalid request. POST an image file."}, status=400)

@csrf_exempt
def feedback(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            
            # Save feedback to database
            feedback = Feedback.objects.create(
                image_url=data.get('image_url'),
                original_detections=json.dumps(data.get('detections')),
                user_feedback=json.dumps(data.get('user_feedback'))
            )
            
            return JsonResponse({
                "status": "success", 
                "feedback_id": feedback.id,
                "message": "Feedback saved successfully"
            })
        except Exception as e:
            return JsonResponse({
                "status": "error",
                "message": str(e)
            }, status=400)
    return JsonResponse({
        "error": "Invalid request method"
    }, status=405)

@csrf_exempt
def get_feedback_samples(request):
    if request.method == "GET":
        try:
            samples = Feedback.objects.all().values(
                'id', 'image_url', 'created_at'
            )[:10]  # Get first 10 samples
            return JsonResponse({
                "status": "success",
                "samples": list(samples)
            })
        except Exception as e:
            return JsonResponse({
                "status": "error",
                "message": str(e)
            }, status=400)
    return JsonResponse({
        "error": "Invalid request method"
    }, status=405)