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


def landing(request):
    """
    Página de presentación/landing principal
    """
    # Contar total de predicciones
    total_predictions = SkinImagePrediction.objects.filter(
        predicted_class__isnull=False
    ).count()
    
    # Obtener últimas 3 predicciones para mostrar
    recent_predictions = SkinImagePrediction.objects.filter(
        predicted_class__isnull=False
    ).order_by('-processed_at')[:3]
    
    context = {
        'total_predictions': total_predictions,
        'recent_predictions': recent_predictions,
        'title': 'DermatologIA - Diagnóstico Inteligente'
    }
    
    return render(request, 'skin_detector/landing.html', context)


def diagnostico(request):
    """
    Página de diagnóstico con formulario de subida (antiguo home)
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


@require_http_methods(["DELETE", "POST"])
def delete_prediction(request, pk):
    """
    Eliminar una predicción del historial
    """
    try:
        prediction = get_object_or_404(SkinImagePrediction, pk=pk)
        
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
