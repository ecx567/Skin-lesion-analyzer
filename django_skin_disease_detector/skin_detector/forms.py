"""
Formularios de Django para la aplicación skin_detector.

Este módulo contiene los formularios utilizados para la validación y procesamiento
de datos de entrada del usuario, siguiendo las mejores prácticas de Django.

Autor: Equipo de Desarrollo DermatologIA
Fecha: Octubre 2025
Versión: 1.0.0
"""

from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from .models import SkinImagePrediction
from .constants import MAX_FILE_SIZE_MB, ALLOWED_IMAGE_EXTENSIONS


# ==================== FORMULARIOS DE AUTENTICACIÓN ====================

class UserRegistrationForm(UserCreationForm):
    """
    Formulario de registro de nuevos usuarios.
    
    Extiende UserCreationForm de Django para incluir campos adicionales
    como email, nombre y apellido, con validaciones mejoradas.
    
    Attributes:
        email (EmailField): Email único del usuario (obligatorio).
        first_name (CharField): Nombre del usuario (obligatorio).
        last_name (CharField): Apellido del usuario (obligatorio).
        
    Validations:
        - Email único en el sistema
        - Contraseña segura (validación de Django)
        - Todos los campos obligatorios
    """
    
    email = forms.EmailField(
        max_length=254,
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Correo electrónico',
            'autocomplete': 'email'
        }),
        help_text='Requerido. Ingrese un correo electrónico válido.'
    )
    
    first_name = forms.CharField(
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Nombre',
            'autocomplete': 'given-name'
        }),
        help_text='Requerido. Ingrese su nombre.'
    )
    
    last_name = forms.CharField(
        max_length=150,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Apellido',
            'autocomplete': 'family-name'
        }),
        help_text='Requerido. Ingrese su apellido.'
    )
    
    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name', 'password1', 'password2']
        
    def __init__(self, *args, **kwargs):
        """Inicializa el formulario con estilos Bootstrap."""
        super().__init__(*args, **kwargs)
        
        # Aplicar clases CSS a todos los campos
        self.fields['username'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Nombre de usuario',
            'autocomplete': 'username'
        })
        self.fields['password1'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Contraseña',
            'autocomplete': 'new-password'
        })
        self.fields['password2'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Confirmar contraseña',
            'autocomplete': 'new-password'
        })
        
        # Textos de ayuda personalizados
        self.fields['username'].help_text = 'Requerido. 150 caracteres o menos. Letras, dígitos y @/./+/-/_ solamente.'
        self.fields['password1'].help_text = 'Mínimo 8 caracteres. No puede ser completamente numérica.'
        self.fields['password2'].help_text = 'Ingrese la misma contraseña para verificación.'
    
    def clean_email(self):
        """
        Valida que el email sea único en el sistema.
        
        Returns:
            str: Email validado.
            
        Raises:
            ValidationError: Si el email ya está registrado.
        """
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError('Este correo electrónico ya está registrado.')
        return email
    
    def save(self, commit=True):
        """
        Guarda el usuario con todos los campos adicionales.
        
        Args:
            commit (bool): Si es True, guarda el usuario en la BD.
            
        Returns:
            User: Instancia del usuario creado.
        """
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        
        if commit:
            user.save()
        return user


class UserLoginForm(AuthenticationForm):
    """
    Formulario de inicio de sesión personalizado.
    
    Extiende AuthenticationForm de Django con estilos mejorados
    y validaciones adicionales.
    
    Attributes:
        username (CharField): Nombre de usuario.
        password (CharField): Contraseña del usuario.
    """
    
    username = forms.CharField(
        max_length=254,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Nombre de usuario',
            'autocomplete': 'username',
            'autofocus': True
        })
    )
    
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Contraseña',
            'autocomplete': 'current-password'
        })
    )
    
    def __init__(self, *args, **kwargs):
        """Inicializa el formulario con configuraciones personalizadas."""
        super().__init__(*args, **kwargs)
        
        # Personalizar mensajes de error
        self.error_messages['invalid_login'] = (
            'Por favor ingrese un nombre de usuario y contraseña correctos. '
            'Ambos campos pueden ser sensibles a mayúsculas.'
        )
        self.error_messages['inactive'] = 'Esta cuenta está inactiva.'


# ==================== FORMULARIOS DE IMÁGENES ====================


class SkinImageUploadForm(forms.ModelForm):
    """
    Formulario para subir y validar imágenes de lesiones cutáneas.
    
    Este formulario maneja la subida de imágenes por parte de los usuarios,
    aplicando validaciones de tamaño, formato y dimensiones mínimas.
    
    Attributes:
        model (Model): Modelo SkinImagePrediction asociado.
        fields (list): Campos del modelo a incluir en el formulario.
        
    Validations:
        - Tamaño máximo: 10MB
        - Formatos permitidos: JPG, JPEG, PNG
        - Dimensiones mínimas: 50x50 píxeles
        
    Methods:
        clean_image: Valida el campo de imagen con reglas personalizadas.
    """
    
    class Meta:
        model = SkinImagePrediction
        fields = ['image']
        
    def __init__(self, *args, **kwargs):
        """
        Inicializa el formulario con atributos HTML personalizados.
        
        Configura la clase CSS, tipos de archivo aceptados y etiquetas
        para mejorar la experiencia del usuario.
        """
        super().__init__(*args, **kwargs)
        
        # Configurar atributos del widget de imagen
        self.fields['image'].widget.attrs.update({
            'class': 'form-control-file',
            'accept': 'image/jpeg,image/jpg,image/png',
            'id': 'imageUpload',
            'aria-describedby': 'imageHelp'
        })
        
        # Etiqueta personalizada
        self.fields['image'].label = 'Seleccionar imagen de lesión cutánea'
        
        # Texto de ayuda
        self.fields['image'].help_text = f'Formatos: {", ".join(ALLOWED_IMAGE_EXTENSIONS).upper()} | Tamaño máximo: {MAX_FILE_SIZE_MB}MB'
        
    def clean_image(self):
        """
        Valida el archivo de imagen subido.
        
        Aplica validaciones de:
        - Tamaño de archivo
        - Extensión/formato
        - Dimensiones mínimas
        
        Returns:
            File: Archivo de imagen validado.
            
        Raises:
            ValidationError: Si el archivo no cumple con los requisitos.
        """
        image = self.cleaned_data.get('image')
        
        if image:
            # Validar tamaño de archivo (máximo configurado en constantes)
            max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
            if image.size > max_size_bytes:
                raise forms.ValidationError(
                    f"El archivo es demasiado grande. Tamaño máximo: {MAX_FILE_SIZE_MB}MB. "
                    f"Su archivo: {image.size / (1024 * 1024):.2f}MB"
                )
            
            # Validar formato/extensión
            extension = image.name.split('.')[-1].lower()
            if extension not in ALLOWED_IMAGE_EXTENSIONS:
                raise forms.ValidationError(
                    f"Formato no válido. Use: {', '.join(ALLOWED_IMAGE_EXTENSIONS).upper()}"
                )
            
            # Validar dimensiones mínimas (si la imagen tiene atributo 'image')
            if hasattr(image, 'image'):
                width, height = image.image.size
                min_dimension = 50
                if width < min_dimension or height < min_dimension:
                    raise forms.ValidationError(
                        f"La imagen es demasiado pequeña. Mínimo {min_dimension}x{min_dimension} píxeles. "
                        f"Su imagen: {width}x{height}"
                    )
                    
        return image


class QuickPredictionForm(forms.Form):
    """
    Formulario simple para predicciones rápidas sin persistencia en BD.
    
    Este formulario se utiliza para realizar predicciones temporales
    sin guardar los datos en la base de datos, ideal para demostraciones
    o pruebas rápidas del sistema.
    
    Attributes:
        image (ImageField): Campo de imagen con validaciones básicas.
        
    Methods:
        clean_image: Valida el campo de imagen.
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
