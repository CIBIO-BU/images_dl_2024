from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from django.conf import settings
import torch
from torchvision import transforms, models
from PIL import Image
import tempfile
import numpy as np
import os
import json
import logging
from .models import Feedback
from ultralytics import YOLO
from PytorchWildlife.models import detection as pw_detection
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)

# Constants
CLASS_NAMES = [
    'aepyceros melampus', 'bos taurus', 'cephalophus nigrifrons', 'crax rubra', 
    'dasyprocta punctata', 'didelphis pernigra', 'equus quagga', 'leopardus pardalis',
    'loxodonta africana', 'madoqua guentheri', 'meleagris ocellata', 'mitu tuberosum',
    'panthera onca', 'pecari tajacu', 'tayassu pecari'
]

CLASSIFICATION_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model Loading (with caching)
@lru_cache(maxsize=1)
def load_yolo_model():
    model_path = os.path.join(settings.BASE_DIR, 'classifier', 'models', 'crop.pt')
    model = YOLO(model_path)
    model.model.eval()
    return model

@lru_cache(maxsize=1)
def load_resnet_model():
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 1024),
        torch.nn.BatchNorm1d(1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(1024, len(CLASS_NAMES)))
    
    checkpoint_path = os.path.join(settings.BASE_DIR, 'classifier', 'models', '15_s.pth')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

@lru_cache(maxsize=1)
def load_md_model():
    return pw_detection.MegaDetectorV6(version="MDV6-yolov9-c")

# Initialize models at startup
yolo_model = load_yolo_model()
resnet = load_resnet_model()
md_model = load_md_model()

def process_detection(cropped_img):
    """Process single detection with ResNet"""
    input_tensor = CLASSIFICATION_TRANSFORM(cropped_img).unsqueeze(0)
    with torch.no_grad():
        output = resnet(input_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        predicted_prob = torch.softmax(output, dim=1)[0, predicted_idx].item()
    return predicted_idx, predicted_prob

@csrf_exempt
def predict(request):
    if request.method != "POST" or not request.FILES.get("file"):
        return JsonResponse({"error": "Invalid request. POST an image file."}, status=400)

    try:
        # Process uploaded file
        with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
            for chunk in request.FILES["file"].chunks():
                temp_file.write(chunk)
            temp_file_path = temp_file.name

            original_image = Image.open(temp_file_path).convert("RGB")
            image_array = np.array(original_image)
            
            # Parallel processing
            with ThreadPoolExecutor() as executor:
                # Run YOLO and MD in parallel
                yolo_future = executor.submit(
                    yolo_model.predict, 
                    source=temp_file_path, 
                    save=False, 
                    imgsz=640, 
                    conf=0.5
                )
                md_future = executor.submit(
                    md_model.single_image_detection,
                    image_array
                )
                
                yolo_results = yolo_future.result()
                md_results = md_future.result()

            detection_results = []
            
            # Process MegaDetector results
            for bbox, label in zip(md_results['detections'], md_results['labels']):
                try:
                    x1, y1, x2, y2 = map(float, bbox[0])
                    cropped_img = original_image.crop((x1, y1, x2, y2))
                    
                    predicted_idx, predicted_prob = process_detection(cropped_img)
                    
                    detection_results.append({
                        "bbox": [x1, y1, x2, y2],
                        "species_prediction": CLASS_NAMES[predicted_idx],
                        "species_confidence": round(float(predicted_prob), 4),
                        "detector": "megadetector"
                    })
                except Exception as e:
                    logger.error(f"Error processing MD detection: {str(e)}")

            # Process YOLO results if needed
            for box in yolo_results[0].boxes:
                try:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    cropped_img = original_image.crop((x1, y1, x2, y2))
                    
                    predicted_idx, predicted_prob = process_detection(cropped_img)
                    
                    detection_results.append({
                        "bbox": [x1, y1, x2, y2],
                        "species_prediction": CLASS_NAMES[predicted_idx],
                        "species_confidence": round(float(box.conf.item()), 4),
                        "detector": "yolo"
                    })
                except Exception as e:
                    logger.error(f"Error processing YOLO detection: {str(e)}")

            return JsonResponse({
                "detections": detection_results,
                "num_detections": len(detection_results)
            })

    except Exception as e:
        logger.exception("Prediction error")
        return JsonResponse({"error": str(e)}, status=500)
    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@csrf_exempt 
def feedback(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = json.loads(request.body)
        
        # Validate feedback data
        required_fields = ['image_url', 'detections', 'user_feedback']
        if not all(field in data for field in required_fields):
            return JsonResponse({"error": "Missing required fields"}, status=400)
        
        # Create feedback with additional metadata
        feedback = Feedback.objects.create(
            image_url=data['image_url'],
            original_detections=json.dumps(data['detections']),
            user_feedback=json.dumps(data['user_feedback']),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
            ip_address=request.META.get('REMOTE_ADDR', '')
        )
        
        # Trigger async model improvement tasks
        from .tasks import process_feedback
        process_feedback.delay(feedback.id)
        
        return JsonResponse({
            "status": "success",
            "feedback_id": feedback.id
        })

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.exception("Feedback submission error")
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def get_feedback_samples(request):
    if request.method != "GET":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        # Get samples with confidence filter
        samples = Feedback.objects.filter(
            original_detections__confidence__gt=0.7
        ).order_by('-created_at').values(
            'id', 'image_url', 'created_at'
        )[:10]
        
        return JsonResponse({
            "status": "success",
            "samples": list(samples)
        })

    except Exception as e:
        logger.exception("Error fetching feedback samples")
        return JsonResponse({"error": str(e)}, status=500)