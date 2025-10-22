"""
Modelos de datos del sistema de detección de enfermedades cutáneas.

Este módulo define los modelos (Models) de la aplicación siguiendo el patrón MTV.
Contiene la estructura de datos para almacenar predicciones de enfermedades cutáneas.

Autor: Equipo de Desarrollo DermatologIA
Fecha: Octubre 2025
Versión: 1.0.0
"""

from django.db import models
from django.core.validators import FileExtensionValidator
import os


class SkinImagePrediction(models.Model):
    """
    Modelo principal para almacenar imágenes de lesiones cutáneas y sus predicciones.
    
    Este modelo representa una predicción individual de enfermedad cutánea,
    almacenando la imagen subida, los resultados del análisis de IA, y metadatos
    relacionados con el procesamiento.
    
    Attributes:
        image (ImageField): Imagen de la lesión cutánea subida por el usuario.
        predicted_class (CharField): Clase predicha por el modelo de IA.
        confidence_score (FloatField): Nivel de confianza de la predicción (0.0-1.0).
        probabilities (JSONField): Dict con probabilidades de todas las 7 clases.
        uploaded_at (DateTimeField): Fecha y hora de subida de la imagen.
        processed_at (DateTimeField): Fecha y hora de procesamiento por la IA.
        image_size (CharField): Dimensiones de la imagen (ej: "800x600").
        processing_time (FloatField): Tiempo de procesamiento en segundos.
        
    Meta:
        ordering: Ordenado por fecha de subida descendente (más reciente primero).
        verbose_name: Nombre singular para el admin de Django.
        verbose_name_plural: Nombre plural para el admin de Django.
        
    Methods:
        __str__: Retorna representación en string del objeto.
        get_disease_display_name: Obtiene el nombre completo de la enfermedad.
        get_confidence_percentage: Retorna la confianza como porcentaje formateado.
        is_high_confidence: Verifica si la predicción tiene alta confianza (>80%).
        get_severity_level: Determina el nivel de severidad de la enfermedad.
    """
    
    # Choices para las 7 clases de enfermedades detectables
    DISEASE_CHOICES = [
        ('akiec', 'Actinic keratoses (Queratosis actínicas)'),
        ('bcc', 'Basal cell carcinoma (Carcinoma basocelular)'),
        ('bkl', 'Benign keratosis (Queratosis benigna)'),
        ('df', 'Dermatofibroma'),
        ('mel', 'Melanoma'),
        ('nv', 'Melanocytic nevi (Nevos melanocíticos)'),
        ('vasc', 'Vascular lesions (Lesiones vasculares)'),
    ]
    
    # Campo de imagen con validación de extensiones permitidas
    image = models.ImageField(
        upload_to='skin_images/',
        validators=[FileExtensionValidator(allowed_extensions=['jpg', 'jpeg', 'png'])],
        help_text="Subir imagen en formato JPG, JPEG o PNG"
    )
    
    # Clase predicha por el modelo de IA
    predicted_class = models.CharField(
        max_length=10,
        choices=DISEASE_CHOICES,
        blank=True,
        null=True,
        help_text="Clase de enfermedad predicha por el modelo"
    )
    
    # Nivel de confianza de la predicción (0.0 a 1.0)
    confidence_score = models.FloatField(
        blank=True,
        null=True,
        help_text="Confianza de la predicción (0-1)"
    )
    
    # Diccionario JSON con probabilidades de todas las 7 clases
    probabilities = models.JSONField(
        blank=True,
        null=True,
        help_text="Probabilidades de todas las clases en formato JSON"
    )
    
    # Timestamps automáticos
    uploaded_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Fecha y hora de subida de la imagen"
    )
    processed_at = models.DateTimeField(
        blank=True,
        null=True,
        help_text="Fecha y hora de procesamiento por la IA"
    )
    
    # Información adicional del procesamiento
    image_size = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        help_text="Dimensiones de la imagen (ancho x alto)"
    )
    processing_time = models.FloatField(
        blank=True,
        null=True,
        help_text="Tiempo de procesamiento en segundos"
    )
    
    class Meta:
        """Configuración de metadatos del modelo."""
        ordering = ['-uploaded_at']  # Ordenar por más reciente primero
        verbose_name = "Predicción de Enfermedad Cutánea"
        verbose_name_plural = "Predicciones de Enfermedades Cutáneas"
        indexes = [
            models.Index(fields=['-uploaded_at'], name='uploaded_at_idx'),
            models.Index(fields=['predicted_class'], name='predicted_class_idx'),
        ]
    
    def __str__(self):
        """
        Representación en string del objeto.
        
        Returns:
            str: Descripción legible del objeto con ID y clase predicha.
        """
        return f"Predicción #{self.id} - {self.predicted_class or 'Sin procesar'} ({self.uploaded_at.strftime('%d/%m/%Y %H:%M')})"
    
    def get_predicted_disease_name(self):
        """
        Obtiene el nombre completo en español de la enfermedad predicha.
        
        Returns:
            str: Nombre completo de la enfermedad o "Sin procesar" si no hay predicción.
            
        Example:
            >>> prediction.predicted_class = 'mel'
            >>> prediction.get_predicted_disease_name()
            'Melanoma'
        """
        if self.predicted_class:
            for code, name in self.DISEASE_CHOICES:
                if code == self.predicted_class:
                    return name
        return "Sin procesar"
    
    def get_confidence_percentage(self):
        """
        Convierte el score de confianza a porcentaje formateado.
        
        Returns:
            float: Nivel de confianza como porcentaje (0-100) redondeado a 2 decimales.
            
        Example:
            >>> prediction.confidence_score = 0.8765
            >>> prediction.get_confidence_percentage()
            87.65
        """
        if self.confidence_score:
            return round(self.confidence_score * 100, 2)
        return 0.0
    
    def is_high_confidence(self, threshold=0.8):
        """
        Verifica si la predicción tiene alta confianza.
        
        Args:
            threshold (float): Umbral de confianza (por defecto 0.8 = 80%).
        
        Returns:
            bool: True si la confianza es mayor o igual al umbral, False en caso contrario.
            
        Example:
            >>> prediction.confidence_score = 0.85
            >>> prediction.is_high_confidence()
            True
        """
        return self.confidence_score >= threshold if self.confidence_score else False
    
    def get_severity_level(self):
        """
        Determina el nivel de severidad de la enfermedad predicha.
        
        Niveles de severidad:
        - 'high': Enfermedades malignas que requieren atención inmediata (mel, bcc, akiec)
        - 'medium': Lesiones que requieren seguimiento (vasc)
        - 'low': Lesiones benignas (bkl, nv, df)
        
        Returns:
            str: Nivel de severidad ('high', 'medium', 'low', o 'unknown').
            
        Example:
            >>> prediction.predicted_class = 'mel'
            >>> prediction.get_severity_level()
            'high'
        """
        severity_map = {
            'mel': 'high',      # Melanoma - Cáncer agresivo
            'bcc': 'high',      # Carcinoma Basocelular - Cáncer
            'akiec': 'high',    # Queratosis Actínica - Precancerosa
            'vasc': 'medium',   # Lesiones Vasculares - Requiere seguimiento
            'bkl': 'low',       # Queratosis Seborreica - Benigna
            'nv': 'low',        # Nevo - Benigno
            'df': 'low',        # Dermatofibroma - Benigno
        }
        return severity_map.get(self.predicted_class, 'unknown')
    
    def delete(self, *args, **kwargs):
        """
        Elimina el objeto y su archivo de imagen asociado del sistema de archivos.
        
        Override del método delete para asegurar que el archivo de imagen
        sea eliminado del disco cuando se borra el registro de la base de datos.
        
        Args:
            *args: Argumentos posicionales para el método delete del padre.
            **kwargs: Argumentos de palabra clave para el método delete del padre.
        """
        # Eliminar archivo físico si existe
        if self.image:
            if os.path.isfile(self.image.path):
                os.remove(self.image.path)
        
        # Llamar al método delete del padre
        super().delete(*args, **kwargs)
