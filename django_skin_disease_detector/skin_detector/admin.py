from django.contrib import admin
from .models import SkinImagePrediction


@admin.register(SkinImagePrediction)
class SkinImagePredictionAdmin(admin.ModelAdmin):
    list_display = [
        'id', 
        'predicted_class', 
        'confidence_percentage_display',
        'uploaded_at', 
        'processed_at',
        'processing_time'
    ]
    
    list_filter = [
        'predicted_class',
        'uploaded_at',
        'processed_at'
    ]
    
    search_fields = ['id', 'predicted_class']
    
    readonly_fields = [
        'uploaded_at', 
        'processed_at', 
        'image_size',
        'processing_time',
        'confidence_score',
        'probabilities'
    ]
    
    def confidence_percentage_display(self, obj):
        if obj.confidence_score:
            return f"{obj.get_confidence_percentage()}%"
        return "-"
    confidence_percentage_display.short_description = "Confianza"
    
    def has_delete_permission(self, request, obj=None):
        # Permitir eliminar registros
        return True
    
    def has_add_permission(self, request):
        # No permitir agregar desde admin (solo desde frontend)
        return False
