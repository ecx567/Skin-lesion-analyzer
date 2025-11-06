from django.urls import path
from . import views
from django.contrib.auth import views as auth_views
from django.urls import reverse_lazy

app_name = 'skin_detector'

urlpatterns = [
    # Autenticación
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    # Google OAuth2
    path('auth/google/', views.google_login, name='google_login'),
    path('auth/google/callback/', views.google_callback, name='google_callback'),
    path('auth/google/unlink/', views.google_unlink, name='google_unlink'),
    path('profile/', views.profile_view, name='profile'),
    
    # Páginas web
    # Mostrar login en la raíz para que el usuario vea primero la pantalla de inicio de sesión
    path('', views.login_view, name='root_login'),
    path('landing/', views.landing, name='landing'),  # Nueva página de presentación (ahora en /landing/)
    path('diagnostico/', views.diagnostico, name='diagnostico'),  # Página de diagnóstico (antiguo home)
    path('disease-info/<str:disease_code>/', views.disease_info, name='disease_info'),  # Info de enfermedades
    path('prediction/<int:pk>/', views.prediction_detail, name='prediction_detail'),
    path('prediction/<int:pk>/pdf/', views.prediction_pdf, name='prediction_pdf'),
    path('history/', views.prediction_history, name='history'),
    path('quick-predict/', views.quick_predict, name='quick_predict'),
    path('save-and-predict/', views.save_and_predict, name='save_and_predict'),
    path('delete/<int:pk>/', views.delete_prediction, name='delete_prediction'),
    
    # API endpoints
    path('api/predict/', views.api_predict, name='api_predict'),
    path('api/info/', views.api_info, name='api_info'),

    # Password reset (secure token-based flow)
    path('password-reset/', auth_views.PasswordResetView.as_view(
        template_name='skin_detector/email/password_reset_form.html',
        email_template_name='skin_detector/email/password_reset_email.txt',
        html_email_template_name='skin_detector/email/password_reset_email.html',
        subject_template_name='skin_detector/email/password_reset_subject.txt',
        success_url=reverse_lazy('skin_detector:password_reset_done')
    ), name='password_reset'),
    path('password-reset/done/', auth_views.PasswordResetDoneView.as_view(
        template_name='skin_detector/email/password_reset_done.html'
    ), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(
        template_name='skin_detector/email/password_reset_confirm.html',
        success_url=reverse_lazy('skin_detector:password_reset_complete')
    ), name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(
        template_name='skin_detector/email/password_reset_complete.html'
    ), name='password_reset_complete'),
]
