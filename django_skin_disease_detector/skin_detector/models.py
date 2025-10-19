from django.db import models
from django.core.validators import FileExtensionValidator
import os


class SkinImagePrediction(models.Model):
    """
    Modelo para almacenar las imágenes subidas y sus predicciones
    """
    DISEASE_CHOICES = [
        ('akiec', 'Actinic keratoses (Queratosis actínicas)'),
        ('bcc', 'Basal cell carcinoma (Carcinoma basocelular)'),
        ('bkl', 'Benign keratosis (Queratosis benigna)'),
        ('df', 'Dermatofibroma'),
        ('mel', 'Melanoma'),
        ('nv', 'Melanocytic nevi (Nevos melanocíticos)'),
        ('vasc', 'Vascular lesions (Lesiones vasculares)'),
    ]
    
    # Imagen subida
    image = models.ImageField(
        upload_to='skin_images/',
        validators=[FileExtensionValidator(allowed_extensions=['jpg', 'jpeg', 'png'])],
        help_text="Subir imagen en formato JPG, JPEG o PNG"
    )
    
    # Resultados de predicción
    predicted_class = models.CharField(
        max_length=10,
        choices=DISEASE_CHOICES,
        blank=True,
        null=True
    )
    
    confidence_score = models.FloatField(
        blank=True,
        null=True,
        help_text="Confianza de la predicción (0-1)"
    )
    
    # Todas las probabilidades de las 7 clases
    probabilities = models.JSONField(
        blank=True,
        null=True,
        help_text="Probabilidades de todas las clases"
    )
    
    # Metadatos
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(blank=True, null=True)
    
    # Información adicional
    image_size = models.CharField(max_length=50, blank=True, null=True)
    processing_time = models.FloatField(blank=True, null=True, help_text="Tiempo en segundos")
    
    class Meta:
        ordering = ['-uploaded_at']
        verbose_name = "Predicción de Enfermedad Cutánea"
        verbose_name_plural = "Predicciones de Enfermedades Cutáneas"
    
    def __str__(self):
        return f"Predicción {self.id} - {self.predicted_class or 'Sin procesar'}"
    
    def get_predicted_disease_name(self):
        """Obtener el nombre completo de la enfermedad predicha"""
        if self.predicted_class:
            for code, name in self.DISEASE_CHOICES:
                if code == self.predicted_class:
                    return name
        return "Sin procesar"
    
    def get_confidence_percentage(self):
        """Obtener la confianza como porcentaje"""
        if self.confidence_score:
            return round(self.confidence_score * 100, 2)
        return 0
    
    def delete(self, *args, **kwargs):
        """Eliminar también el archivo de imagen al borrar el registro"""
        if self.image:
            if os.path.isfile(self.image.path):
                os.remove(self.image.path)
        super().delete(*args, **kwargs)
