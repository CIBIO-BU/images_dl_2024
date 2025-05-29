from django.db import models
import json

class Feedback(models.Model):
    image_url = models.CharField(max_length=500)
    original_detections = models.TextField()  # Stores JSON as text
    user_feedback = models.TextField()  # Stores JSON as text
    created_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)  # For tracking retraining
    
    def get_original_detections(self):
        return json.loads(self.original_detections)
    
    def get_user_feedback(self):
        return json.loads(self.user_feedback)
    
    def __str__(self):
        return f"Feedback #{self.id} for {self.image_url}"
    
    class Meta:
        ordering = ['-created_at']