from django.urls import path
from . import views

app_name = 'skin_detector'

urlpatterns = [
    # Páginas web
    path('', views.home, name='home'),
    path('prediction/<int:pk>/', views.prediction_detail, name='prediction_detail'),
    path('history/', views.prediction_history, name='history'),
    path('quick-predict/', views.quick_predict, name='quick_predict'),
    
    # API endpoints
    path('api/predict/', views.api_predict, name='api_predict'),
    path('api/info/', views.api_info, name='api_info'),
]
