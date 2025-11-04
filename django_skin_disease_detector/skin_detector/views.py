"""
Vistas del sistema de detección de enfermedades cutáneas.

Este módulo contiene todas las vistas (Views) de la aplicación siguiendo el patrón MTV.
Incluye vistas web para usuarios y endpoints API REST.

Autor: Equipo de Desarrollo DermatologIA
Fecha: Octubre 2025
Versión: 1.0.0
"""

from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.http import JsonResponse
from django.contrib import messages
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
import json
import os
import time
import requests
import uuid
from .models import SkinImagePrediction, SocialAccount
from .forms import SkinImageUploadForm, QuickPredictionForm, UserRegistrationForm, UserLoginForm
from .predictor import get_predictor
import logging
from django.template.loader import render_to_string
from django.http import HttpResponse
from django.db.utils import OperationalError
from django.conf import settings
from django.contrib.staticfiles import finders
from django.contrib.auth.forms import PasswordChangeForm, SetPasswordForm
from django.contrib.auth import update_session_auth_hash

# Configurar logger para esta aplicación
logger = logging.getLogger(__name__)


# ==================== VISTAS DE AUTENTICACIÓN ====================

def register_view(request):
    """
    Vista de registro de nuevos usuarios.
    
    Maneja el registro de usuarios nuevos con validación de datos
    y creación automática de cuenta.
    
    Args:
        request (HttpRequest): Objeto de solicitud HTTP.
        
    Returns:
        HttpResponse: Renderiza formulario de registro o redirige al login.
        
    Template:
        skin_detector/register.html
        
    Context:
        form (UserRegistrationForm): Formulario de registro.
        title (str): Título de la página.
    """
    if request.user.is_authenticated:
        messages.info(request, 'Ya has iniciado sesión.')
        return redirect('skin_detector:diagnostico')
    
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            messages.success(
                request, 
                f'¡Cuenta creada exitosamente para {username}! Ahora puedes iniciar sesión.'
            )
            logger.info(f'Nuevo usuario registrado: {username}')
            return redirect('skin_detector:login')
        else:
            messages.error(request, 'Por favor corrige los errores en el formulario.')
    else:
        form = UserRegistrationForm()
    
    context = {
        'form': form,
        'title': 'Registro de Usuario - DermatologIA'
    }
    return render(request, 'skin_detector/register.html', context)


def google_login(request):
    """
    Inicia flujo OAuth2 con Google: redirige a la página de autorización.
    Requiere configurar GOOGLE_OAUTH2_CLIENT_ID y GOOGLE_OAUTH2_REDIRECT_URI en settings.
    """
    client_id = getattr(settings, 'GOOGLE_OAUTH2_CLIENT_ID', None)
    redirect_uri = request.build_absolute_uri(reverse('skin_detector:google_callback'))
    if not client_id:
        messages.error(request, 'Google OAuth no configurado (GOOGLE_OAUTH2_CLIENT_ID faltante).')
        return redirect('skin_detector:login')

    state = str(uuid.uuid4())
    request.session['google_oauth_state'] = state

    auth_url = (
        'https://accounts.google.com/o/oauth2/v2/auth'
        f'?response_type=code&client_id={client_id}'
        f'&redirect_uri={redirect_uri}'
        '&scope=openid%20email%20profile'
        f'&state={state}&access_type=online&prompt=select_account'
    )

    return redirect(auth_url)


def google_callback(request):
    """
    Callback que Google redirige con `code`. Intercambia code por token, obtiene userinfo,
    crea/obtiene usuario local y hace login.
    """
    error = request.GET.get('error')
    if error:
        messages.error(request, f'Google OAuth error: {error}')
        return redirect('skin_detector:login')

    code = request.GET.get('code')
    state = request.GET.get('state')
    session_state = request.session.pop('google_oauth_state', None)
    if not code or not state or state != session_state:
        messages.error(request, 'Estado de OAuth inválido o código faltante.')
        return redirect('skin_detector:login')

    client_id = getattr(settings, 'GOOGLE_OAUTH2_CLIENT_ID', None)
    client_secret = getattr(settings, 'GOOGLE_OAUTH2_CLIENT_SECRET', None)
    redirect_uri = request.build_absolute_uri(reverse('skin_detector:google_callback'))

    if not client_id or not client_secret:
        messages.error(request, 'Google OAuth no configurado (cliente/secret faltantes).')
        return redirect('skin_detector:login')

    token_endpoint = 'https://oauth2.googleapis.com/token'
    try:
        token_resp = requests.post(token_endpoint, data={
            'code': code,
            'client_id': client_id,
            'client_secret': client_secret,
            'redirect_uri': redirect_uri,
            'grant_type': 'authorization_code'
        }, timeout=10)
        token_resp.raise_for_status()
        token_data = token_resp.json()
        access_token = token_data.get('access_token')
    except Exception as e:
        messages.error(request, f'Error intercambiando token con Google: {e}')
        return redirect('skin_detector:login')

    # Obtener información del usuario
    try:
        userinfo_resp = requests.get('https://openidconnect.googleapis.com/v1/userinfo', headers={
            'Authorization': f'Bearer {access_token}'
        }, timeout=10)
        userinfo_resp.raise_for_status()
        userinfo = userinfo_resp.json()
    except Exception as e:
        messages.error(request, f'Error obteniendo información de usuario: {e}')
        return redirect('skin_detector:login')
    email = userinfo.get('email')
    name = userinfo.get('name') or ''
    provider_user_id = userinfo.get('sub')
    email_verified = userinfo.get('email_verified', False)
    if not email:
        messages.error(request, 'No se pudo obtener correo electrónico de Google.')
        return redirect('skin_detector:login')

    # Crear o obtener usuario local
    from django.contrib.auth.models import User

    try:
        # Si el usuario ya tiene una SocialAccount con este provider_user_id, usarla
        if provider_user_id:
            sa = SocialAccount.objects.filter(provider='google', provider_user_id=provider_user_id).first()
        else:
            sa = None

        # Si el usuario está autenticado, estamos en flujo de vinculación
        if request.user.is_authenticated:
            if not provider_user_id:
                messages.error(request, 'No se pudo obtener identificación de Google para vincular.')
                return redirect('skin_detector:diagnostico')

            # Si existe y pertenece a otro usuario, denegar
            if sa and sa.user != request.user:
                messages.error(request, 'Esta cuenta de Google ya está vinculada a otra cuenta.')
                return redirect('skin_detector:diagnostico')

            if sa and sa.user == request.user:
                messages.info(request, 'Tu cuenta de Google ya está vinculada.')
                return redirect('skin_detector:diagnostico')

            # Requerir correo verificado para vincular
            if not email_verified:
                messages.error(request, 'Para vincular la cuenta de Google, el correo debe estar verificado en Google.')
                return redirect('skin_detector:diagnostico')

            # Crear SocialAccount
            SocialAccount.objects.create(
                user=request.user,
                provider='google',
                provider_user_id=provider_user_id,
                email=email
            )
            messages.success(request, 'Cuenta de Google vinculada correctamente.')
            return redirect('skin_detector:diagnostico')

        # No autenticado: intentar login por SocialAccount
        if sa:
            login(request, sa.user)
            messages.success(request, f'Has iniciado sesión como {sa.user.username} usando Google.')
            return redirect('skin_detector:diagnostico')

        # No hay SocialAccount: buscar usuario por email
        user = User.objects.filter(email__iexact=email).first()
        if user:
            # Auto-vincular si correo verificado
            if provider_user_id and email_verified:
                try:
                    SocialAccount.objects.create(
                        user=user,
                        provider='google',
                        provider_user_id=provider_user_id,
                        email=email
                    )
                except Exception:
                    # Ignorar si hay conflicto de unicidad
                    pass

            login(request, user)
            messages.success(request, f'Has iniciado sesión como {user.username} usando Google.')
            return redirect('skin_detector:diagnostico')

        # Crear nuevo usuario y vincular
        base_username = email.split('@')[0]
        username = base_username
        counter = 1
        while User.objects.filter(username=username).exists():
            username = f"{base_username}{counter}"
            counter += 1

        user = User.objects.create_user(username=username, email=email)
        if ' ' in name:
            first, *rest = name.split(' ')
            user.first_name = first
            user.last_name = ' '.join(rest)
        else:
            user.first_name = name
        user.set_unusable_password()
        user.save()

        if provider_user_id:
            try:
                SocialAccount.objects.create(
                    user=user,
                    provider='google',
                    provider_user_id=provider_user_id,
                    email=email
                )
            except Exception:
                pass
        login(request, user)
        messages.success(request, f'Cuenta creada y autenticada como {user.username} (Google).')
        return redirect('skin_detector:diagnostico')

        login(request, user)
        messages.success(request, f'Cuenta creada y autenticada como {user.username} (Google).')
        return redirect('skin_detector:diagnostico')

    except Exception as e:
        logger.exception('Error en flujo Google OAuth')
        messages.error(request, f'Error creando o accediendo al usuario: {e}')
        return redirect('skin_detector:login')


@login_required
def google_unlink(request):
    """Desvincula la cuenta de Google del usuario autenticado si existe."""
    try:
        sa = SocialAccount.objects.filter(user=request.user, provider='google').first()
        if not sa:
            messages.info(request, 'No hay ninguna cuenta de Google vinculada a tu perfil.')
            return redirect('skin_detector:diagnostico')

        # Comprobar que el usuario tenga otra forma de autenticación
        has_password = request.user.has_usable_password()
        has_other_social = SocialAccount.objects.filter(user=request.user).exclude(provider='google').exists()
        if not has_password and not has_other_social:
            messages.error(
                request,
                'No puedes desvincular Google porque no tienes otra forma de inicio de sesión. ' 
                'Por favor establece una contraseña en tu perfil o añade otro método de acceso antes de desvincular.'
            )
            return redirect('skin_detector:diagnostico')

        sa.delete()
        messages.success(request, 'Cuenta de Google desvinculada correctamente.')
        return redirect('skin_detector:diagnostico')
    except Exception as e:
        logger.exception('Error desvinculando cuenta Google')
        messages.error(request, f'Error desvinculando cuenta: {e}')
        return redirect('skin_detector:diagnostico')


@login_required
def profile_view(request):
    """Página de perfil donde el usuario puede ver su info y establecer/cambiar contraseña.

    - Si el usuario tiene contraseña usable, se muestra PasswordChangeForm (requiere la contraseña actual).
    - Si no tiene contraseña usable (p. ej. creado solo por Google), se muestra SetPasswordForm para crear una.
    """
    user = request.user
    if user.has_usable_password():
        FormClass = PasswordChangeForm
    else:
        FormClass = SetPasswordForm

    if request.method == 'POST':
        form = FormClass(user, request.POST) if FormClass is PasswordChangeForm else FormClass(user, request.POST)
        if form.is_valid():
            form.save()
            # Mantener la sesión válida después de cambiar la contraseña
            update_session_auth_hash(request, user)
            messages.success(request, 'Contraseña establecida/cambiada correctamente.')
            return redirect('skin_detector:profile')
        else:
            messages.error(request, 'Corrige los errores en el formulario.')
    else:
        form = FormClass(user) if FormClass is PasswordChangeForm else FormClass(user)

    context = {
        'title': 'Mi Perfil',
        'form': form,
        'user_obj': user
    }
    return render(request, 'skin_detector/profile.html', context)


def login_view(request):
    """
    Vista de inicio de sesión.
    
    Autentica usuarios existentes y redirige a la página de diagnóstico.
    
    Args:
        request (HttpRequest): Objeto de solicitud HTTP.
        
    Returns:
        HttpResponse: Renderiza formulario de login o redirige al diagnóstico.
        
    Template:
        skin_detector/login.html
        
    Context:
        form (UserLoginForm): Formulario de inicio de sesión.
        title (str): Título de la página.
    """
    if request.user.is_authenticated:
        messages.info(request, 'Ya has iniciado sesión.')
        return redirect('skin_detector:diagnostico')
    
    if request.method == 'POST':
        form = UserLoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            
            if user is not None:
                login(request, user)
                messages.success(request, f'¡Bienvenido de nuevo, {username}!')
                logger.info(f'Usuario autenticado: {username}')
                
                # Redirigir a la página solicitada o al diagnóstico
                next_page = request.GET.get('next', 'skin_detector:diagnostico')
                return redirect(next_page)
            else:
                messages.error(request, 'Usuario o contraseña incorrectos.')
        else:
            messages.error(request, 'Usuario o contraseña incorrectos.')
    else:
        form = UserLoginForm()
    
    context = {
        'form': form,
        'title': 'Iniciar Sesión - DermatologIA'
    }
    return render(request, 'skin_detector/login.html', context)


@login_required
def logout_view(request):
    """
    Vista de cierre de sesión.
    
    Cierra la sesión del usuario actual y redirige a la landing page.
    
    Args:
        request (HttpRequest): Objeto de solicitud HTTP.
        
    Returns:
        HttpResponseRedirect: Redirige a la página de inicio.
    """
    username = request.user.username
    logout(request)
    messages.info(request, f'Has cerrado sesión correctamente. ¡Hasta pronto, {username}!')
    logger.info(f'Usuario cerró sesión: {username}')
    return redirect('skin_detector:landing')


# ==================== VISTAS WEB ====================

def landing(request):
    """
    Vista de página de presentación/landing principal.
    
    Muestra la página de inicio del sistema con estadísticas generales,
    información de las enfermedades detectables y acceso rápido al diagnóstico.
    
    Args:
        request (HttpRequest): Objeto de solicitud HTTP de Django.
    
    Returns:
        HttpResponse: Renderiza la plantilla landing.html con contexto de datos.
        
    Template:
        skin_detector/landing.html
        
    Context:
        total_predictions (int): Número total de predicciones realizadas.
        recent_predictions (QuerySet): Últimas 3 predicciones exitosas.
        title (str): Título de la página.
    """
    # Si el usuario no está autenticado, no mostrar estadísticas ni historial (cada usuario tiene su propio historial)
    if not request.user.is_authenticated:
        total_predictions = 0
        recent_predictions = []
    else:
        try:
            # Contar total de predicciones del usuario autenticado
            total_predictions = SkinImagePrediction.objects.filter(
                predicted_class__isnull=False,
                user=request.user
            ).count()

            # Obtener últimas 3 predicciones del usuario
            recent_predictions = SkinImagePrediction.objects.filter(
                predicted_class__isnull=False,
                user=request.user
            ).order_by('-processed_at')[:3]
        except OperationalError:
            # Esquema antiguo (sin columna user); usar comportamiento global para evitar 500
            total_predictions = SkinImagePrediction.objects.filter(predicted_class__isnull=False).count()
            recent_predictions = SkinImagePrediction.objects.filter(predicted_class__isnull=False).order_by('-processed_at')[:3]
    
    context = {
        'total_predictions': total_predictions,
        'recent_predictions': recent_predictions,
        'title': 'DermatologIA - Diagnóstico Inteligente'
    }
    
    return render(request, 'skin_detector/landing.html', context)


@login_required
def diagnostico(request):
    """
    Vista de página de diagnóstico con formulario de subida de imágenes.
    
    Maneja tanto GET (mostrar formulario) como POST (procesar imagen y realizar predicción).
    Utiliza el modelo de IA para clasificar la lesión cutánea subida por el usuario.
    
    Args:
        request (HttpRequest): Objeto de solicitud HTTP de Django.
    
    Returns:
        HttpResponse: 
            - GET: Renderiza formulario de subida de imagen.
            - POST: Redirige a página de detalles de predicción o muestra errores.
    
    Template:
        skin_detector/home.html
        
    Context:
        form (SkinImageUploadForm): Formulario de subida de imagen.
        recent_predictions (QuerySet): Últimas 5 predicciones para mostrar.
        title (str): Título de la página.
        
    Raises:
        Exception: Captura y registra cualquier error durante la predicción.
    """
    if request.method == 'POST':
        form = SkinImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Guardar imagen (sin commit para poder asignar usuario si corresponde)
            prediction_obj = form.save(commit=False)
            if request.user.is_authenticated:
                prediction_obj.user = request.user
            # Guardar para tener el archivo disponible en disco
            prediction_obj.save()

            try:
                # Realizar predicción
                predictor = get_predictor()
                result = predictor.predict(prediction_obj.image.path)

                # Actualizar objeto con resultados
                prediction_obj.predicted_class = result['predicted_class']
                prediction_obj.confidence_score = result['confidence']
                prediction_obj.probabilities = result['all_probabilities']
                prediction_obj.processing_time = result.get('processing_time')
                prediction_obj.processed_at = timezone.now()

                # Obtener dimensiones de la imagen
                from PIL import Image
                with Image.open(prediction_obj.image.path) as img:
                    prediction_obj.image_size = f"{img.size[0]}x{img.size[1]}"

                prediction_obj.save()

                messages.success(request, 'Imagen procesada exitosamente!')
                return redirect('skin_detector:prediction_detail', pk=prediction_obj.pk)

            except Exception as e:
                logger.error(f"Error en predicción: {str(e)}")
                messages.error(request, f'Error procesando imagen: {str(e)}')
                try:
                    prediction_obj.delete()  # Limpiar imagen si falla
                except Exception:
                    pass
                
    else:
        form = SkinImageUploadForm()
    
    # Obtener últimas predicciones
    try:
        if request.user.is_authenticated:
            # Mostrar solo las últimas predicciones del usuario autenticado
            recent_predictions = SkinImagePrediction.objects.filter(
                predicted_class__isnull=False,
                user=request.user
            ).order_by('-processed_at')[:5]
        else:
            recent_predictions = SkinImagePrediction.objects.filter(
                predicted_class__isnull=False
            ).order_by('-processed_at')[:5]
    except OperationalError:
        # Si la columna user no existe u otro error de esquema, caer a versión global
        recent_predictions = SkinImagePrediction.objects.filter(predicted_class__isnull=False).order_by('-processed_at')[:5]
    
    context = {
        'form': form,
        'recent_predictions': recent_predictions,
        'title': 'Detector de Enfermedades Cutáneas'
    }
    
    return render(request, 'skin_detector/home.html', context)


@login_required
def prediction_detail(request, pk):
    """
    Detalle de una predicción específica
    """
    try:
        prediction = get_object_or_404(SkinImagePrediction, pk=pk)

        # Verificar propiedad: si la predicción tiene usuario y no coincide, denegar acceso
        if prediction.user and prediction.user != request.user:
            return HttpResponse('No autorizado', status=403)
    except OperationalError:
        # Esquema desactualizado: indicar al administrador que ejecute migraciones
        return HttpResponse('Error: esquema de base de datos desactualizado. Ejecute las migraciones (manage.py migrate).', status=500)
    
    # Obtener predictor para información adicional
    try:
        predictor = get_predictor()
        
        # Si no está procesada, procesarla ahora
        if not prediction.predicted_class and prediction.image:
            result = predictor.predict(prediction.image.path)
            
            prediction.predicted_class = result['predicted_class']
            prediction.confidence_score = result['confidence']
            prediction.probabilities = result['all_probabilities']
            prediction.processing_time = result['processing_time']
            prediction.processed_at = timezone.now()
            prediction.save()
        
        # Obtener top 3 predicciones si está procesada
        top_predictions = None
        if prediction.probabilities:
            sorted_probs = sorted(
                prediction.probabilities.items(),
                key=lambda x: x[1]['probability'],
                reverse=True
            )[:3]
            
            top_predictions = []
            for i, (class_code, prob_info) in enumerate(sorted_probs):
                disease_info = predictor.disease_info.get(class_code, {})
                top_predictions.append({
                    'rank': i + 1,
                    'class_code': class_code,
                    'percentage': prob_info['percentage'],
                    'name': prob_info['name'],
                    'spanish': prob_info['spanish'],
                    'disease_info': disease_info
                })
                
    except Exception as e:
        logger.error(f"Error obteniendo detalles: {str(e)}")
        messages.error(request, f'Error: {str(e)}')
        top_predictions = None
    
    context = {
        'prediction': prediction,
        'top_predictions': top_predictions,
        'title': f'Predicción #{prediction.pk}'
    }
    
    return render(request, 'skin_detector/prediction_detail.html', context)


@login_required
def prediction_history(request):
    """
    Historial de todas las predicciones
    """
    # Mostrar solo predicciones del usuario autenticado
    predictions = SkinImagePrediction.objects.filter(user=request.user).order_by('-uploaded_at')
    
    context = {
        'predictions': predictions,
        'title': 'Historial de Predicciones'
    }
    
    return render(request, 'skin_detector/history.html', context)


@login_required
def prediction_pdf(request, pk):
    """
    Genera un PDF con el reporte de la predicción.
    """
    try:
        prediction = get_object_or_404(SkinImagePrediction, pk=pk)

        # Verificar propiedad: si la predicción tiene usuario y no coincide, denegar acceso
        if prediction.user and prediction.user != request.user:
            return HttpResponse('No autorizado', status=403)
    except OperationalError:
        return HttpResponse('Error: esquema de base de datos desactualizado. Ejecute las migraciones (manage.py migrate).', status=500)

    # Preparar contexto similar a prediction_detail
    top_predictions = None
    try:
        predictor = get_predictor()
        if prediction.probabilities:
            sorted_probs = sorted(
                prediction.probabilities.items(),
                key=lambda x: x[1]['probability'],
                reverse=True
            )[:3]
            top_predictions = []
            for i, (class_code, prob_info) in enumerate(sorted_probs):
                disease_info = predictor.disease_info.get(class_code, {})
                top_predictions.append({
                    'rank': i + 1,
                    'class_code': class_code,
                    'percentage': prob_info['percentage'],
                    'name': prob_info['name'],
                    'spanish': prob_info['spanish'],
                    'disease_info': disease_info
                })
    except Exception:
        top_predictions = None

    context = {
        'prediction': prediction,
        'top_predictions': top_predictions,
        'title': f'Reporte Predicción #{prediction.pk}'
    }

    # Renderizar la plantilla a HTML
    html = render_to_string('skin_detector/prediction_report.html', context)

    # Importación perezosa de xhtml2pdf (pisa) para evitar que una importación fallida a nivel de módulo
    # deje el nombre en None en procesos que ya estaban corriendo.
    try:
        from xhtml2pdf import pisa
    except Exception:
        return HttpResponse('La generación de PDF no está disponible. Instale xhtml2pdf.', status=500)

    # Generar PDF
    result = HttpResponse(content_type='application/pdf')
    result['Content-Disposition'] = f'attachment; filename=prediction_{prediction.pk}.pdf'

    # Función para que xhtml2pdf resuelva rutas de static y media
    def link_callback(uri, rel):
        # Media files
        if uri.startswith(settings.MEDIA_URL):
            path = os.path.join(settings.MEDIA_ROOT, uri.replace(settings.MEDIA_URL, ''))
            return path

        # Static files
        if uri.startswith(settings.STATIC_URL):
            static_path = uri.replace(settings.STATIC_URL, '')
            result_path = finders.find(static_path)
            if result_path:
                return result_path

        # Fallback a la URI
        return uri

    pisa_status = pisa.CreatePDF(src=html, dest=result, link_callback=link_callback)

    if pisa_status.err:
        return HttpResponse('Error generando PDF: ' + str(pisa_status.err), status=500)

    return result


@csrf_exempt
@login_required
def quick_predict(request):
    """
    Predicción rápida sin guardar en base de datos
    """
    if request.method == 'POST':
        form = QuickPredictionForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                image = form.cleaned_data['image']
                
                # Guardar temporalmente la imagen
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    for chunk in image.chunks():
                        tmp_file.write(chunk)
                    temp_path = tmp_file.name
                
                # Realizar predicción
                predictor = get_predictor()
                result = predictor.get_top_predictions(temp_path, top_n=3)
                
                # Limpiar archivo temporal
                os.unlink(temp_path)
                
                return JsonResponse({
                    'success': True,
                    'result': result
                })
                
            except Exception as e:
                logger.error(f"Error en predicción rápida: {str(e)}")
                return JsonResponse({
                    'success': False,
                    'error': str(e)
                }, status=500)
        else:
            return JsonResponse({
                'success': False,
                'error': 'Formulario inválido',
                'form_errors': form.errors
            }, status=400)
    
    return JsonResponse({'error': 'Método no permitido'}, status=405)


@require_http_methods(['POST'])
@login_required
def save_and_predict(request):
    """Guarda la imagen enviada, ejecuta la predicción y devuelve URL del detalle."""
    form = SkinImageUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        return JsonResponse({'success': False, 'error': 'Formulario inválido', 'form_errors': form.errors}, status=400)

    prediction_obj = None
    try:
        # Guardar sin commit para asignar usuario si está autenticado
        prediction_obj = form.save(commit=False)
        if request.user.is_authenticated:
            prediction_obj.user = request.user
        prediction_obj.save()

        # Realizar predicción
        predictor = get_predictor()
        result = predictor.predict(prediction_obj.image.path)

        # Actualizar objeto con resultados
        prediction_obj.predicted_class = result['predicted_class']
        prediction_obj.confidence_score = result['confidence']
        prediction_obj.probabilities = result['all_probabilities']
        prediction_obj.processing_time = result.get('processing_time')
        prediction_obj.processed_at = timezone.now()

        # Obtener dimensiones de la imagen
        from PIL import Image
        with Image.open(prediction_obj.image.path) as img:
            prediction_obj.image_size = f"{img.size[0]}x{img.size[1]}"

        prediction_obj.save()

        redirect_url = request.build_absolute_uri(reverse('skin_detector:prediction_detail', args=[prediction_obj.pk]))
        return JsonResponse({'success': True, 'redirect_url': redirect_url})

    except Exception as e:
        logger.exception('Error saving and predicting image')
        # Intentar limpiar archivo si fue creado
        try:
            if prediction_obj:
                prediction_obj.delete()
        except Exception:
            pass
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


# ==================== API REST ====================

@api_view(['POST'])
@permission_classes([AllowAny])
@login_required
def api_predict(request):
    """
    API endpoint para predicción
    """
    try:
        if 'image' not in request.FILES:
            return Response({
                'error': 'No se proporcionó imagen'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        image_file = request.FILES['image']
        
        # Validaciones básicas
        if image_file.size > 10 * 1024 * 1024:  # 10MB
            return Response({
                'error': 'Imagen demasiado grande (máx 10MB)'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Guardar temporalmente
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            for chunk in image_file.chunks():
                tmp_file.write(chunk)
            temp_path = tmp_file.name
        
        # Realizar predicción
        predictor = get_predictor()
        result = predictor.get_top_predictions(temp_path, top_n=5)
        
        # Limpiar archivo temporal
        os.unlink(temp_path)
        
        # Respuesta API
        api_response = {
            'success': True,
            'prediction': {
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence_percentage'],
                'class_name': result['class_name_spanish'],
                'severity': result['disease_info']['severity'],
                'recommendation': result['disease_info']['recommendation'],
                'processing_time': result['processing_time']
            },
            'top_predictions': result['top_predictions'],
            'all_probabilities': result['all_probabilities']
        }
        
        return Response(api_response, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error en API: {str(e)}")
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([AllowAny])
@login_required
def api_info(request):
    """
    Información sobre las clases que puede detectar el modelo
    """
    try:
        predictor = get_predictor()
        
        classes_info = {}
        for idx, class_data in predictor.class_names.items():
            class_code = class_data['code']
            classes_info[class_code] = {
                'name': class_data['name'],
                'spanish': class_data['spanish'],
                'severity': predictor.disease_info[class_code]['severity'],
                'description': predictor.disease_info[class_code]['description']
            }
        
        return Response({
            'model_classes': classes_info,
            'total_classes': len(classes_info),
            'model_info': {
                'input_size': f"{predictor.img_size}x{predictor.img_size}",
                'supported_formats': ['JPG', 'JPEG', 'PNG'],
                'max_file_size': '10MB'
            }
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@require_http_methods(["DELETE", "POST"])
@login_required
def delete_prediction(request, pk):
    """
    Eliminar una predicción del historial
    """
    try:
        try:
            prediction = get_object_or_404(SkinImagePrediction, pk=pk)

            # Verificar propiedad: solo el propietario puede eliminar
            if prediction.user and prediction.user != request.user:
                return JsonResponse({'success': False, 'error': 'No autorizado'}, status=403)
        except OperationalError:
            return JsonResponse({'success': False, 'error': 'Esquema de base de datos desactualizado. Ejecute las migraciones.'}, status=500)
        
        # Eliminar archivo de imagen si existe
        if prediction.image:
            if os.path.exists(prediction.image.path):
                os.remove(prediction.image.path)
        
        # Eliminar el registro de la base de datos
        prediction.delete()
        
        return JsonResponse({
            'success': True,
            'message': 'Predicción eliminada correctamente'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)


def disease_info(request, disease_code):
    """
    Página de información detallada de cada enfermedad
    """
    # Diccionario con información completa de cada enfermedad
    DISEASE_DATA = {
        'mel': {
            'code': 'MEL',
            'full_name': 'Melanoma',
            'icon': '⚫',
            'color_bg': '#fee2e2',
            'color_text': '#991b1b',
            'description': 'El melanoma es el tipo más grave de cáncer de piel. Se desarrolla en las células (melanocitos) que producen melanina, el pigmento que da color a la piel.',
            'symptoms': [
                'Lunares nuevos o cambios en lunares existentes',
                'Manchas oscuras asimétricas con bordes irregulares',
                'Lesiones que cambian de tamaño, forma o color',
                'Sangrado o picazón en un lunar',
                'Lesiones con múltiples colores (marrón, negro, rojo, azul)',
            ],
            'locations': ['Cualquier parte del cuerpo', 'Espalda', 'Piernas', 'Brazos', 'Rostro'],
            'zones': ['Piel expuesta al sol', 'Áreas con lunares', 'Piel clara'],
            'treatments': [
                'Cirugía para extirpar el melanoma',
                'Inmunoterapia para estimular el sistema inmunológico',
                'Terapia dirigida para mutaciones genéticas específicas',
                'Radioterapia en casos avanzados',
                'Quimioterapia para melanomas metastásicos',
            ],
            'prevention': [
                'Evitar exposición prolongada al sol, especialmente entre 10 AM y 4 PM',
                'Usar protector solar SPF 30+ diariamente',
                'Usar ropa protectora (sombreros, camisas de manga larga)',
                'Evitar camas de bronceado',
                'Realizar autoexámenes mensuales de la piel',
                'Revisiones dermatológicas anuales',
            ],
            'severity': 'high',
            'alert_message': 'El melanoma es un cáncer agresivo que requiere atención médica inmediata. Si detectas cambios en lunares o manchas, consulta a un dermatólogo urgentemente.',
        },
        'bcc': {
            'code': 'BCC',
            'full_name': 'Carcinoma Basocelular',
            'icon': '⚠️',
            'color_bg': '#fed7aa',
            'color_text': '#9a3412',
            'description': 'El carcinoma basocelular es el tipo más común de cáncer de piel. Crece lentamente y rara vez se propaga a otras partes del cuerpo, pero puede ser invasivo localmente.',
            'symptoms': [
                'Protuberancia perlada o cerosa',
                'Lesión plana de color carne o marrón',
                'Llaga con sangrado o costra que cicatriza y vuelve',
                'Área blanca similar a una cicatriz',
                'Borde enrollado con centro deprimido',
            ],
            'locations': ['Rostro', 'Cuello', 'Orejas', 'Cuero cabelludo', 'Hombros'],
            'zones': ['Áreas expuestas al sol', 'Cabeza', 'Cuello'],
            'treatments': [
                'Extirpación quirúrgica (cirugía de Mohs)',
                'Curetaje y electrodesecación',
                'Crioterapia (congelación)',
                'Cremas tópicas (imiquimod, 5-fluorouracilo)',
                'Radioterapia en casos no quirúrgicos',
            ],
            'prevention': [
                'Protección solar constante',
                'Evitar exposición solar en horas pico',
                'Usar sombreros de ala ancha',
                'Revisiones dermatológicas regulares',
                'Proteger cicatrices de exposición solar',
            ],
            'severity': 'medium',
            'alert_message': 'Aunque crece lentamente, el carcinoma basocelular debe tratarse para evitar daño extenso al tejido circundante. Consulta a un dermatólogo para evaluación.',
        },
        'akiec': {
            'code': 'AKIEC',
            'full_name': 'Queratosis Actínica / Carcinoma Intraepitelial',
            'icon': '🔥',
            'color_bg': '#fef3c7',
            'color_text': '#92400e',
            'description': 'La queratosis actínica es una lesión precancerosa causada por daño solar crónico. Puede progresar a carcinoma de células escamosas si no se trata.',
            'symptoms': [
                'Parches ásperos y escamosos en la piel',
                'Superficie seca o con costra',
                'Color rosa, rojo o marrón',
                'Textura como papel de lija',
                'Sensación de ardor o picazón',
            ],
            'locations': ['Rostro', 'Labios', 'Orejas', 'Dorso de manos', 'Antebrazos', 'Cuero cabelludo'],
            'zones': ['Piel con daño solar', 'Áreas expuestas crónicamente'],
            'treatments': [
                'Crioterapia (nitrógeno líquido)',
                'Cremas tópicas (imiquimod, diclofenaco, 5-FU)',
                'Terapia fotodinámica',
                'Curetaje y cauterización',
                'Peelings químicos',
                'Tratamiento láser',
            ],
            'prevention': [
                'Uso diario de protector solar SPF 50+',
                'Evitar exposición solar innecesaria',
                'Usar ropa protectora',
                'Revisiones dermatológicas cada 6 meses',
                'Tratar lesiones tempranamente',
            ],
            'severity': 'medium',
            'alert_message': 'Las queratosis actínicas son lesiones precancerosas que deben tratarse para prevenir su progresión a cáncer de piel. Consulta a un dermatólogo.',
        },
        'bkl': {
            'code': 'BKL',
            'full_name': 'Queratosis Seborreica',
            'icon': '🟤',
            'color_bg': '#e9d5ff',
            'color_text': '#581c87',
            'description': 'La queratosis seborreica es una lesión cutánea benigna muy común. Aparece como crecimientos elevados de color marrón, negro o tostado que parecen "pegados" a la piel.',
            'symptoms': [
                'Crecimientos elevados con apariencia verrugosa',
                'Color marrón, negro o amarillento',
                'Superficie con textura cerosa o escamosa',
                'Apariencia de "pegados" a la piel',
                'Múltiples lesiones en algunas personas',
            ],
            'locations': ['Rostro', 'Pecho', 'Espalda', 'Hombros', 'Cuero cabelludo'],
            'zones': ['Tronco', 'Extremidades', 'Cabeza'],
            'treatments': [
                'No requiere tratamiento (benigno)',
                'Crioterapia si es cosméticamente molesto',
                'Curetaje para remoción',
                'Electrodesecación',
                'Ablación láser',
            ],
            'prevention': [
                'No se puede prevenir (parte del envejecimiento)',
                'Protección solar general',
                'Evitar irritación de las lesiones',
                'Consultar si hay cambios o crecimiento rápido',
            ],
            'severity': 'low',
            'alert_message': 'La queratosis seborreica es completamente benigna y no requiere tratamiento. Solo se remueve por razones estéticas o si causa irritación.',
        },
        'nv': {
            'code': 'NV',
            'full_name': 'Nevo Melanocítico (Lunar)',
            'icon': '⭕',
            'color_bg': '#d1fae5',
            'color_text': '#065f46',
            'description': 'Los nevos melanocíticos, comúnmente llamados lunares, son crecimientos benignos de melanocitos. La mayoría son inofensivos, pero algunos pueden transformarse en melanoma.',
            'symptoms': [
                'Manchas o protuberancias redondas u ovaladas',
                'Color uniforme (marrón, negro, rosa)',
                'Bordes bien definidos',
                'Tamaño generalmente menor a 6mm',
                'Pueden ser planos o elevados',
            ],
            'locations': ['Cualquier parte del cuerpo', 'Rostro', 'Tronco', 'Extremidades'],
            'zones': ['Todo el cuerpo', 'Áreas con exposición solar'],
            'treatments': [
                'Observación regular (regla ABCDE)',
                'Extirpación quirúrgica si hay cambios sospechosos',
                'Biopsia para evaluación histológica',
                'Fotografía de seguimiento',
            ],
            'prevention': [
                'Protección solar para prevenir nuevos lunares',
                'Autoexamen mensual (regla ABCDE)',
                'Revisión dermatológica anual',
                'Fotografiar lunares para comparación',
                'Evitar camas de bronceado',
            ],
            'severity': 'low',
            'alert_message': 'Los lunares son generalmente benignos, pero deben monitorearse. Consulta a un dermatólogo si observas cambios en tamaño, forma, color o si aparecen síntomas.',
        },
        'vasc': {
            'code': 'VASC',
            'full_name': 'Lesiones Vasculares',
            'icon': '❤️',
            'color_bg': '#fecaca',
            'color_text': '#7f1d1d',
            'description': 'Las lesiones vasculares son crecimientos o malformaciones de vasos sanguíneos en la piel. Incluyen hemangiomas, angiomas, telangiectasias y otras condiciones vasculares.',
            'symptoms': [
                'Manchas rojas o púrpuras en la piel',
                'Protuberancias de color rojo brillante',
                'Vasos sanguíneos visibles (arañas vasculares)',
                'Pueden blanquear al presionarlos',
                'Varían desde planas hasta elevadas',
            ],
            'locations': ['Rostro', 'Cuello', 'Pecho', 'Extremidades', 'Cualquier zona'],
            'zones': ['Piel', 'Mucosas', 'Áreas expuestas'],
            'treatments': [
                'Láser vascular (láser de colorante pulsado)',
                'Escleroterapia para vasos pequeños',
                'Electrocoagulación',
                'Crioterapia en casos específicos',
                'Observación si es asintomático',
            ],
            'prevention': [
                'Protección solar',
                'Evitar traumatismos',
                'Control de condiciones subyacentes',
                'Cuidado de la piel adecuado',
            ],
            'severity': 'low',
            'alert_message': 'La mayoría de lesiones vasculares son benignas y cosméticamente tratables. Consulta a un dermatólogo si crecen rápidamente o causan molestias.',
        },
        'df': {
            'code': 'DF',
            'full_name': 'Dermatofibroma',
            'icon': '🔘',
            'color_bg': '#e5e7eb',
            'color_text': '#1f2937',
            'description': 'El dermatofibroma es un nódulo cutáneo benigno común. Es una proliferación de fibroblastos que generalmente aparece después de un traumatismo menor o picadura de insecto.',
            'symptoms': [
                'Nódulo firme al tacto',
                'Color marrón, rojo o púrpura',
                'Se hunde ligeramente al pellizcar (signo del hoyuelo)',
                'Generalmente indoloro',
                'Crece lentamente',
            ],
            'locations': ['Piernas', 'Brazos', 'Tronco'],
            'zones': ['Extremidades inferiores', 'Brazos'],
            'treatments': [
                'No requiere tratamiento (benigno)',
                'Extirpación quirúrgica si es sintomático',
                'Crioterapia superficial',
                'Inyección de corticoides',
            ],
            'prevention': [
                'No se puede prevenir',
                'Evitar traumatismos repetidos',
                'No manipular las lesiones',
            ],
            'severity': 'low',
            'alert_message': 'El dermatofibroma es completamente benigno y generalmente no requiere tratamiento. Solo se remueve si causa molestias o por razones estéticas.',
        },
    }
    
    # Obtener datos de la enfermedad
    disease_code_lower = disease_code.lower()
    disease_data = DISEASE_DATA.get(disease_code_lower)
    
    if not disease_data:
        # Si no existe la enfermedad, redirigir al landing
        messages.error(request, 'Enfermedad no encontrada.')
        return redirect('skin_detector:landing')
    
    context = {
        'disease_code': disease_code_lower,
        'disease_name': disease_data['full_name'],
        'disease_data': disease_data,
        'title': f'{disease_data["full_name"]} - Información Detallada'
    }
    
    return render(request, 'skin_detector/disease_info.html', context)
