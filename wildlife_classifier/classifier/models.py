# app/models.py
from django.db import models

class Image(models.Model):
    image_file = models.ImageField(upload_to="images/%Y/%m/%d/")
    original_filename = models.CharField(max_length=255, blank=True)
    width = models.PositiveIntegerField(null=True, blank=True)
    height = models.PositiveIntegerField(null=True, blank=True)
    sha256 = models.CharField(max_length=64, unique=True, blank=True)  # optional de-dupe
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Image {self.id} ({self.original_filename})"


class Detection(models.Model):
    image = models.ForeignKey(Image, on_delete=models.CASCADE, related_name="detections")
    # [x_min, y_min, x_max, y_max] in original image pixel coords
    bbox = models.JSONField()
    predicted_label = models.CharField(max_length=128)
    predicted_confidence = models.FloatField()
    index = models.PositiveIntegerField()  # stable index per image

    class Meta:
        unique_together = ("image", "index")  # one index per image

    def __str__(self):
        return f"Det {self.id} on Image {self.image_id} ({self.predicted_label})"


class Feedback(models.Model):
    image = models.ForeignKey(Image, on_delete=models.CASCADE, related_name="feedbacks")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Feedback {self.id} on Image {self.image_id}"


class FeedbackAnnotation(models.Model):
    feedback = models.ForeignKey(Feedback, on_delete=models.CASCADE, related_name="annotations")
    detection = models.ForeignKey(Detection, on_delete=models.SET_NULL, null=True, blank=True)
    index = models.PositiveIntegerField()  # mirrors Detection.index (useful if detection missing)
    bbox_correct = models.BooleanField(default=True)
    species_correct = models.BooleanField(default=True)
    correct_species = models.CharField(max_length=128, blank=True)

    class Meta:
        unique_together = ("feedback", "index")

    def __str__(self):
        return f"FB ann (fb={self.feedback_id}, idx={self.index})"
