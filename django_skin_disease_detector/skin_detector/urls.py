from django.urls import path
from . import views

app_name = 'skin_detector'

urlpatterns = [
    # Autenticación
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    
    # Páginas web
    path('', views.landing, name='landing'),  # Nueva página de presentación
    path('diagnostico/', views.diagnostico, name='diagnostico'),  # Página de diagnóstico (antiguo home)
    path('disease-info/<str:disease_code>/', views.disease_info, name='disease_info'),  # Info de enfermedades
    path('prediction/<int:pk>/', views.prediction_detail, name='prediction_detail'),
    path('history/', views.prediction_history, name='history'),
    path('quick-predict/', views.quick_predict, name='quick_predict'),
    path('delete/<int:pk>/', views.delete_prediction, name='delete_prediction'),
    
    # API endpoints
    path('api/predict/', views.api_predict, name='api_predict'),
    path('api/info/', views.api_info, name='api_info'),
]
