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
import logging
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification
from django.core.exceptions import ValidationError

logger = logging.getLogger(__name__)

# Load models
md_model = pw_detection.MegaDetectorV6(version="MDV6-yolov10-e")

# md_classifier = pw_classification.AI4GSerengeti()

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

        md_results = md_model.single_image_detection(image_array)

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
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method is allowed"}, status=405)
    
    try:
        # For multipart/form-data, get data from request.FILES and request.POST
        image_file = request.FILES.get("file")
        detections_json = request.POST.get("detections")
        user_feedback_json = request.POST.get("user_feedback")

        if not image_file or not detections_json or not user_feedback_json:
            return JsonResponse({
                "error": "Missing required fields. Required: file, detections, user_feedback"
            }, status=400)

        # Parse JSON fields
        try:
            detections = json.loads(detections_json)
            user_feedback = json.loads(user_feedback_json)
        except Exception as e:
            return JsonResponse({"error": f"Invalid JSON in fields: {str(e)}"}, status=400)

        # Save image to media/feedback_images/
        from .models import Feedback
        feedback = Feedback(
            original_detections=detections,
            user_feedback=user_feedback
        )
        feedback.image.save(image_file.name, image_file)
        feedback.full_clean()
        feedback.save()

        return JsonResponse({
            "status": "success",
            "feedback_id": feedback.id,
            "message": "Feedback saved successfully",
            "created_at": feedback.created_at.isoformat()
        }, status=201)

    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        logger.exception("Error saving feedback")
        return JsonResponse({"error": "Internal server error"}, status=500)
    
@csrf_exempt
def get_feedback_samples(request):
    if request.method == "GET":
        try:
            samples = Feedback.objects.all().order_by('-created_at').values(
                'id', 'image_url', 'created_at', 'original_detections'
            )[:10]  # Get 10 most recent samples
            
            # Process samples to include detection count
            sample_list = []
            for sample in samples:
                sample_data = {
                    'id': sample['id'],
                    'image_url': sample['image_url'],
                    'created_at': sample['created_at'],
                    'detection_count': len(sample['original_detections']) if sample['original_detections'] else 0
                }
                sample_list.append(sample_data)
            
            return JsonResponse({
                "status": "success",
                "samples": sample_list
            })
        except Exception as e:
            return JsonResponse({
                "status": "error",
                "message": str(e)
            }, status=500)
    return JsonResponse({
        "error": "Invalid request method"
    }, status=405)

@csrf_exempt
def get_feedback_detail(request, feedback_id):
    if request.method == "GET":
        try:
            feedback = Feedback.objects.get(id=feedback_id)
            
            response_data = {
                'id': feedback.id,
                'image_url': feedback.image_url,
                'created_at': feedback.created_at,
                'original_detections': feedback.original_detections,
                'user_feedback': feedback.user_feedback
            }
            
            return JsonResponse({
                "status": "success",
                "feedback": response_data
            })
        except Feedback.DoesNotExist:
            return JsonResponse({
                "status": "error",
                "message": "Feedback not found"
            }, status=404)
        except Exception as e:
            return JsonResponse({
                "status": "error",
                "message": str(e)
            }, status=500)
    return JsonResponse({
        "error": "Invalid request method"
    }, status=405)