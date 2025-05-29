from django.urls import path, include
from .views import predict, feedback, get_feedback_samples

urlpatterns = [
    path("predict/", predict, name="predict"),
    path("feedback/", feedback, name="feedback"),
    path("feedback_samples/", get_feedback_samples, name="get_feedback_samples"),
]