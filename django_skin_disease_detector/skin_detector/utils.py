"""
Funciones de utilidad para la aplicación skin_detector.

Este módulo contiene funciones auxiliares reutilizables que siguen el principio DRY
y facilitan el mantenimiento del código.

Autor: Equipo de Desarrollo DermatologIA
Fecha: Octubre 2025
Versión: 1.0.0
"""

from PIL import Image
import os
from django.core.files.uploadedfile import UploadedFile
from .constants import (
    ALLOWED_IMAGE_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    DISEASE_NAMES,
    SEVERITY_LEVELS,
)


def validate_image_file(file: UploadedFile) -> tuple[bool, str]:
    """
    Valida que un archivo subido sea una imagen válida.
    
    Verifica la extensión del archivo y el tamaño máximo permitido.
    
    Args:
        file (UploadedFile): Archivo subido por el usuario.
    
    Returns:
        tuple[bool, str]: (es_valido, mensaje_error)
            - es_valido: True si el archivo es válido, False en caso contrario.
            - mensaje_error: Descripción del error o cadena vacía si es válido.
    
    Example:
        >>> is_valid, error = validate_image_file(uploaded_file)
        >>> if not is_valid:
        ...     print(f"Error: {error}")
    """
    # Verificar que el archivo existe
    if not file:
        return False, "No se ha subido ningún archivo."
    
    # Obtener extensión del archivo
    file_extension = file.name.split('.')[-1].lower()
    
    # Verificar extensión permitida
    if file_extension not in ALLOWED_IMAGE_EXTENSIONS:
        return False, f"Tipo de archivo no válido. Solo se permiten: {', '.join(ALLOWED_IMAGE_EXTENSIONS).upper()}"
    
    # Verificar tamaño del archivo
    if file.size > MAX_FILE_SIZE_BYTES:
        max_size_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        return False, f"El archivo es demasiado grande. Tamaño máximo: {max_size_mb}MB"
    
    return True, ""


def get_image_dimensions(image_path: str) -> tuple[int, int]:
    """
    Obtiene las dimensiones de una imagen.
    
    Args:
        image_path (str): Ruta al archivo de imagen.
    
    Returns:
        tuple[int, int]: (ancho, alto) de la imagen en píxeles.
        Retorna (0, 0) si hay error al abrir la imagen.
    
    Example:
        >>> width, height = get_image_dimensions('/path/to/image.jpg')
        >>> print(f"Dimensiones: {width}x{height}")
    """
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return 0, 0


def format_image_size(width: int, height: int) -> str:
    """
    Formatea las dimensiones de una imagen como string.
    
    Args:
        width (int): Ancho en píxeles.
        height (int): Alto en píxeles.
    
    Returns:
        str: Dimensiones formateadas como "WIDTHxHEIGHT".
    
    Example:
        >>> format_image_size(800, 600)
        '800x600'
    """
    return f"{width}x{height}"


def get_disease_full_name(disease_code: str) -> str:
    """
    Obtiene el nombre completo de una enfermedad dado su código.
    
    Args:
        disease_code (str): Código de la enfermedad (ej: 'mel', 'bcc').
    
    Returns:
        str: Nombre completo de la enfermedad o el código si no se encuentra.
    
    Example:
        >>> get_disease_full_name('mel')
        'Melanoma'
    """
    return DISEASE_NAMES.get(disease_code.lower(), disease_code.upper())


def get_disease_severity(disease_code: str) -> str:
    """
    Obtiene el nivel de severidad de una enfermedad.
    
    Args:
        disease_code (str): Código de la enfermedad.
    
    Returns:
        str: Nivel de severidad ('high', 'medium', 'low', 'unknown').
    
    Example:
        >>> get_disease_severity('mel')
        'high'
    """
    return SEVERITY_LEVELS.get(disease_code.lower(), 'unknown')


def format_confidence_score(score: float) -> str:
    """
    Formatea un score de confianza como porcentaje.
    
    Args:
        score (float): Score de confianza entre 0 y 1.
    
    Returns:
        str: Porcentaje formateado con 2 decimales y símbolo %.
    
    Example:
        >>> format_confidence_score(0.8765)
        '87.65%'
    """
    percentage = score * 100
    return f"{percentage:.2f}%"


def format_processing_time(seconds: float) -> str:
    """
    Formatea el tiempo de procesamiento en un formato legible.
    
    Args:
        seconds (float): Tiempo en segundos.
    
    Returns:
        str: Tiempo formateado (ej: "2.35s" o "125ms").
    
    Example:
        >>> format_processing_time(2.345)
        '2.35s'
        >>> format_processing_time(0.125)
        '125ms'
    """
    if seconds < 1:
        milliseconds = seconds * 1000
        return f"{milliseconds:.0f}ms"
    return f"{seconds:.2f}s"


def safe_file_delete(file_path: str) -> bool:
    """
    Elimina un archivo de forma segura verificando que existe.
    
    Args:
        file_path (str): Ruta al archivo a eliminar.
    
    Returns:
        bool: True si el archivo fue eliminado, False si no existía o hubo error.
    
    Example:
        >>> safe_file_delete('/path/to/file.jpg')
        True
    """
    try:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            os.remove(file_path)
            return True
        return False
    except Exception:
        return False


def get_confidence_level(score: float) -> str:
    """
    Determina el nivel de confianza en formato texto.
    
    Args:
        score (float): Score de confianza entre 0 y 1.
    
    Returns:
        str: Nivel de confianza ('Muy Alta', 'Alta', 'Media', 'Baja').
    
    Example:
        >>> get_confidence_level(0.95)
        'Muy Alta'
        >>> get_confidence_level(0.65)
        'Media'
    """
    if score >= 0.9:
        return 'Muy Alta'
    elif score >= 0.8:
        return 'Alta'
    elif score >= 0.6:
        return 'Media'
    else:
        return 'Baja'


def truncate_filename(filename: str, max_length: int = 50) -> str:
    """
    Trunca un nombre de archivo largo manteniendo la extensión.
    
    Args:
        filename (str): Nombre completo del archivo.
        max_length (int): Longitud máxima del nombre (default: 50).
    
    Returns:
        str: Nombre de archivo truncado con extensión preservada.
    
    Example:
        >>> truncate_filename('very_long_filename_here.jpg', 20)
        'very_long_f...e.jpg'
    """
    if len(filename) <= max_length:
        return filename
    
    name, extension = os.path.splitext(filename)
    max_name_length = max_length - len(extension) - 3  # 3 por '...'
    
    if max_name_length <= 0:
        return filename[:max_length]
    
    return f"{name[:max_name_length]}...{extension}"


def sort_probabilities_desc(probabilities: dict) -> list:
    """
    Ordena las probabilidades de mayor a menor.
    
    Args:
        probabilities (dict): Diccionario con {clase: probabilidad}.
    
    Returns:
        list: Lista de tuplas (clase, probabilidad) ordenadas descendentemente.
    
    Example:
        >>> probs = {'mel': 0.85, 'bcc': 0.10, 'nv': 0.05}
        >>> sort_probabilities_desc(probs)
        [('mel', 0.85), ('bcc', 0.10), ('nv', 0.05)]
    """
    return sorted(probabilities.items(), key=lambda x: x[1], reverse=True)


def calculate_confidence_color(score: float) -> str:
    """
    Retorna un color CSS basado en el nivel de confianza.
    
    Args:
        score (float): Score de confianza entre 0 y 1.
    
    Returns:
        str: Nombre de clase CSS de Bootstrap para el color.
    
    Example:
        >>> calculate_confidence_color(0.95)
        'success'
        >>> calculate_confidence_color(0.50)
        'warning'
    """
    if score >= 0.8:
        return 'success'  # Verde
    elif score >= 0.6:
        return 'info'     # Azul
    elif score >= 0.4:
        return 'warning'  # Amarillo
    else:
        return 'danger'   # Rojo
