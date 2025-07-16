from django.db import models
from django.core.validators import URLValidator

class Feedback(models.Model):
    image_url = models.TextField(validators=[URLValidator()])
    original_detections = models.JSONField()
    user_feedback = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    def clean(self):
        """Custom validation"""
        super().clean()
        if not isinstance(self.original_detections, list):
            raise ValidationError("original_detections must be a list")
        if not isinstance(self.user_feedback, dict):
            raise ValidationError("user_feedback must be a dictionary")

    def __str__(self):
        return f"Feedback #{self.id} for {self.image_url}"