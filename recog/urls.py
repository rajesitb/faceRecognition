from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('capture/', views.capture_cadet_image),
    path('attendance/', views.take_attendance),
]
