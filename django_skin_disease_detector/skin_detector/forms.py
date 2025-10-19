from django import forms
from .models import SkinImagePrediction


class SkinImageUploadForm(forms.ModelForm):
    """
    Formulario para subir imágenes de enfermedades cutáneas
    """
    
    class Meta:
        model = SkinImagePrediction
        fields = ['image']
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['image'].widget.attrs.update({
            'class': 'form-control-file',
            'accept': 'image/jpeg,image/jpg,image/png',
            'id': 'imageUpload'
        })
        self.fields['image'].label = 'Seleccionar imagen de lesión cutánea'
        
    def clean_image(self):
        image = self.cleaned_data.get('image')
        
        if image:
            # Validar tamaño de archivo (máximo 10MB)
            if image.size > 10 * 1024 * 1024:
                raise forms.ValidationError("El archivo es demasiado grande. Máximo 10MB.")
            
            # Validar formato
            valid_extensions = ['jpg', 'jpeg', 'png']
            extension = image.name.split('.')[-1].lower()
            if extension not in valid_extensions:
                raise forms.ValidationError("Formato no válido. Use JPG, JPEG o PNG.")
            
            # Validar dimensiones mínimas
            if hasattr(image, 'image'):
                width, height = image.image.size
                if width < 50 or height < 50:
                    raise forms.ValidationError("La imagen es demasiado pequeña. Mínimo 50x50 píxeles.")
                    
        return image


class QuickPredictionForm(forms.Form):
    """
    Formulario simple para predicciones rápidas sin guardar en BD
    """
    image = forms.ImageField(
        label='Imagen para análisis rápido',
        widget=forms.FileInput(attrs={
            'class': 'form-control-file',
            'accept': 'image/jpeg,image/jpg,image/png',
            'id': 'quickImageUpload'
        })
    )
    
    def clean_image(self):
        image = self.cleaned_data.get('image')
        
        if image:
            # Validaciones básicas
            if image.size > 10 * 1024 * 1024:
                raise forms.ValidationError("Archivo muy grande. Máximo 10MB.")
                
            valid_extensions = ['jpg', 'jpeg', 'png']
            extension = image.name.split('.')[-1].lower()
            if extension not in valid_extensions:
                raise forms.ValidationError("Use formato JPG, JPEG o PNG.")
                
        return image
