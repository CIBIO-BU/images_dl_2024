from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict, name='predict'),
    path('feedback/', views.feedback, name='feedback'),
    path('feedback/samples/', views.get_feedback_samples, name='feedback-samples'),
    path('feedback/<int:feedback_id>/', views.get_feedback_detail, name='feedback-detail'),
]