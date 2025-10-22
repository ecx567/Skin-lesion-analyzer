"""
Configuración y constantes de la aplicación skin_detector.

Este módulo centraliza todas las constantes, configuraciones y datos estáticos
utilizados en la aplicación para facilitar el mantenimiento y seguir el principio DRY.

Autor: Equipo de Desarrollo DermatologIA
Fecha: Octubre 2025
Versión: 1.0.0
"""

# ==================== CONFIGURACIÓN DEL MODELO DE IA ====================

# Ruta al modelo entrenado
MODEL_PATH = 'models/improved_balanced_7class_model.h5'

# Dimensiones de entrada del modelo
IMAGE_SIZE = (28, 28)

# Número de clases que puede predecir el modelo
NUM_CLASSES = 7

# Umbral de confianza para considerar una predicción como "alta confianza"
HIGH_CONFIDENCE_THRESHOLD = 0.8

# Umbral de confianza mínima para mostrar la predicción
MIN_CONFIDENCE_THRESHOLD = 0.5


# ==================== CLASES DE ENFERMEDADES ====================

# Códigos de enfermedades detectables
DISEASE_CODES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Mapeo de códigos a nombres completos
DISEASE_NAMES = {
    'akiec': 'Queratosis Actínica / Carcinoma Intraepitelial',
    'bcc': 'Carcinoma Basocelular',
    'bkl': 'Queratosis Seborreica',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Nevo Melanocítico (Lunar)',
    'vasc': 'Lesiones Vasculares',
}

# Nombres completos en inglés (para compatibilidad)
DISEASE_NAMES_EN = {
    'akiec': 'Actinic keratoses and intraepithelial carcinoma',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions',
}

# Nivel de severidad por enfermedad
SEVERITY_LEVELS = {
    'mel': 'high',      # Melanoma - Cáncer agresivo
    'bcc': 'high',      # Carcinoma Basocelular - Cáncer
    'akiec': 'high',    # Queratosis Actínica - Precancerosa
    'vasc': 'medium',   # Lesiones Vasculares - Requiere seguimiento
    'bkl': 'low',       # Queratosis Seborreica - Benigna
    'nv': 'low',        # Nevo - Benigno
    'df': 'low',        # Dermatofibroma - Benigno
}

# Descripciones cortas de cada enfermedad
DISEASE_DESCRIPTIONS = {
    'mel': 'Tipo más grave de cáncer de piel que se desarrolla en los melanocitos. Requiere atención urgente.',
    'bcc': 'Tipo más común de cáncer de piel. Crece lentamente y rara vez se propaga.',
    'akiec': 'Lesión precancerosa causada por daño solar crónico. Puede progresar a cáncer.',
    'bkl': 'Crecimiento cutáneo benigno muy común. No requiere tratamiento.',
    'nv': 'Lunar común benigno. La mayoría son inofensivos pero deben monitorearse.',
    'vasc': 'Crecimientos o malformaciones de vasos sanguíneos. Generalmente benignos.',
    'df': 'Nódulo cutáneo benigno común. Completamente benigno.',
}

# Iconos para cada enfermedad
DISEASE_ICONS = {
    'mel': '⚫',
    'bcc': '⚠️',
    'akiec': '🔥',
    'bkl': '🟤',
    'nv': '⭕',
    'vasc': '❤️',
    'df': '🔘',
}

# Colores de fondo para cada enfermedad
DISEASE_COLORS = {
    'mel': {'bg': '#fee2e2', 'text': '#991b1b'},
    'bcc': {'bg': '#fed7aa', 'text': '#9a3412'},
    'akiec': {'bg': '#fef3c7', 'text': '#92400e'},
    'bkl': {'bg': '#e9d5ff', 'text': '#581c87'},
    'nv': {'bg': '#d1fae5', 'text': '#065f46'},
    'vasc': {'bg': '#fecaca', 'text': '#7f1d1d'},
    'df': {'bg': '#e5e7eb', 'text': '#1f2937'},
}


# ==================== CONFIGURACIÓN DE ARCHIVOS ====================

# Extensiones de archivo permitidas para subida
ALLOWED_IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png']

# Tamaño máximo de archivo en MB
MAX_FILE_SIZE_MB = 10

# Tamaño máximo de archivo en bytes
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Directorio de subida de imágenes
UPLOAD_DIR = 'skin_images/'


# ==================== MENSAJES DE USUARIO ====================

# Mensajes de éxito
SUCCESS_MESSAGES = {
    'prediction_success': 'Imagen procesada exitosamente!',
    'deletion_success': 'Predicción eliminada correctamente.',
}

# Mensajes de error
ERROR_MESSAGES = {
    'invalid_file_type': 'Tipo de archivo no válido. Solo se permiten imágenes JPG, JPEG y PNG.',
    'file_too_large': f'El archivo es demasiado grande. Máximo permitido: {MAX_FILE_SIZE_MB}MB.',
    'prediction_error': 'Error al procesar la imagen. Por favor, intente nuevamente.',
    'model_not_found': 'Modelo de IA no encontrado. Contacte al administrador.',
    'invalid_image': 'La imagen no es válida o está corrupta.',
}

# Mensajes de advertencia
WARNING_MESSAGES = {
    'low_confidence': 'La confianza de la predicción es baja. Se recomienda consultar con un especialista.',
    'high_severity': 'La lesión detectada requiere atención médica inmediata.',
}


# ==================== CONFIGURACIÓN DE API ====================

# Versión de la API
API_VERSION = 'v1'

# Rate limiting (peticiones por minuto)
API_RATE_LIMIT = 60

# Timeout para procesamiento de predicciones (segundos)
PREDICTION_TIMEOUT = 30


# ==================== DESCARGO DE RESPONSABILIDAD ====================

MEDICAL_DISCLAIMER = """
Este sistema es una herramienta de apoyo y no reemplaza la consulta médica profesional.
Siempre consulte con un dermatólogo para obtener un diagnóstico definitivo y plan de
tratamiento personalizado.
"""

# Versión corta del descargo
MEDICAL_DISCLAIMER_SHORT = "Este sistema es solo de apoyo. Consulte siempre con un dermatólogo."


# ==================== INFORMACIÓN DEL SISTEMA ====================

# Versión de la aplicación
APP_VERSION = '1.0.0'

# Nombre de la aplicación
APP_NAME = 'DermatologIA'

# Descripción de la aplicación
APP_DESCRIPTION = 'Sistema inteligente de diagnóstico dermatológico utilizando IA'

# Información del modelo
MODEL_INFO = {
    'name': 'Improved Balanced 7-Class CNN Model',
    'framework': 'TensorFlow/Keras',
    'dataset': 'HAM10000',
    'classes': NUM_CLASSES,
    'accuracy': '92%',  # Actualizar con la precisión real del modelo
}
