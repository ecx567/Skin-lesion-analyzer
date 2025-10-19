from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.contrib import messages
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
from .models import SkinImagePrediction
from .forms import SkinImageUploadForm, QuickPredictionForm
from .predictor import get_predictor
import logging

logger = logging.getLogger(__name__)


def home(request):
    """
    Página principal con formulario de subida
    """
    if request.method == 'POST':
        form = SkinImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Guardar imagen
            prediction_obj = form.save()
            
            try:
                # Realizar predicción
                predictor = get_predictor()
                result = predictor.predict(prediction_obj.image.path)
                
                # Actualizar objeto con resultados
                prediction_obj.predicted_class = result['predicted_class']
                prediction_obj.confidence_score = result['confidence']
                prediction_obj.probabilities = result['all_probabilities']
                prediction_obj.processing_time = result['processing_time']
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
                prediction_obj.delete()  # Limpiar imagen si falla
                
    else:
        form = SkinImageUploadForm()
    
    # Obtener últimas predicciones
    recent_predictions = SkinImagePrediction.objects.filter(
        predicted_class__isnull=False
    ).order_by('-processed_at')[:5]
    
    context = {
        'form': form,
        'recent_predictions': recent_predictions,
        'title': 'Detector de Enfermedades Cutáneas'
    }
    
    return render(request, 'skin_detector/home.html', context)


def prediction_detail(request, pk):
    """
    Detalle de una predicción específica
    """
    prediction = get_object_or_404(SkinImagePrediction, pk=pk)
    
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


def prediction_history(request):
    """
    Historial de todas las predicciones
    """
    predictions = SkinImagePrediction.objects.all().order_by('-uploaded_at')
    
    context = {
        'predictions': predictions,
        'title': 'Historial de Predicciones'
    }
    
    return render(request, 'skin_detector/history.html', context)


@csrf_exempt
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


# ==================== API REST ====================

@api_view(['POST'])
@permission_classes([AllowAny])
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
