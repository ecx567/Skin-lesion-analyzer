# 🏥 Django Skin Disease Detector - Sistema de Detección de Enfermedades Cutáneas

## 📋 Índice
- [Descripción General](#descripción-general)
- [Arquitectura del Proyecto](#arquitectura-del-proyecto)
- [Estructura de Archivos](#estructura-de-archivos)
- [Modelos (Models)](#modelos-models)
- [Vistas (Views)](#vistas-views)
- [Formularios (Forms)](#formularios-forms)
- [URLs y Rutas](#urls-y-rutas)
- [Templates](#templates)
- [Sistema de Autenticación](#sistema-de-autenticación)
- [API REST](#api-rest)
- [Base de Datos](#base-de-datos)
- [Instalación y Configuración](#instalación-y-configuración)
- [Uso del Sistema](#uso-del-sistema)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)

---

## 🎯 Descripción General

**Django Skin Disease Detector** es una aplicación web completa para la detección de enfermedades cutáneas mediante inteligencia artificial. Utiliza un modelo de Deep Learning entrenado con el dataset HAM10000 para clasificar lesiones de piel en 7 categorías diferentes.

### Características Principales:

✅ **Detección de IA**: Clasifica 7 tipos de enfermedades cutáneas con >85% de precisión
✅ **Sistema de Autenticación**: Registro, login y gestión de usuarios
✅ **Historial de Diagnósticos**: Almacena todas las predicciones del usuario
✅ **API REST**: Endpoints para integración con aplicaciones móviles
✅ **Base de Datos Supabase**: PostgreSQL en la nube con sincronización
✅ **Interfaz Moderna**: Diseño responsive con Bootstrap 5
✅ **Información Detallada**: Descripciones completas de cada enfermedad
✅ **Resultados Visuales**: Gráficas de probabilidades y niveles de confianza

---

## 🏗️ Arquitectura del Proyecto

### Patrón MTV (Model-Template-View)

Django utiliza el patrón MTV, una variante del MVC:

```
┌─────────────────────────────────────────────────────────────┐
│                     ARQUITECTURA MTV                         │
└─────────────────────────────────────────────────────────────┘

    USUARIO (Navegador Web / App Móvil)
         │
         │ HTTP Request (GET/POST)
         ▼
┌─────────────────────────────────────────────────────────────┐
│  URLS.PY (URLconf - Enrutador)                              │
│  • Mapea URLs a vistas                                      │
│  • skin_detector/urls.py + skin_disease_project/urls.py    │
└─────────────────────────────────────────────────────────────┘
         │
         │ Llama a la vista correspondiente
         ▼
┌─────────────────────────────────────────────────────────────┐
│  VIEWS.PY (Lógica de Negocio)                               │
│  • Procesa requests                                         │
│  • Interactúa con modelos                                   │
│  • Ejecuta predicciones de IA                               │
│  • Prepara contexto para templates                          │
└─────────────────────────────────────────────────────────────┘
         │                                │
         │ Consulta/Guarda               │ Renderiza
         ▼                                ▼
┌──────────────────────────┐   ┌────────────────────────────┐
│  MODELS.PY               │   │  TEMPLATES/                │
│  • Define estructura     │   │  • HTML + Django Template  │
│    de datos              │   │    Language (DTL)          │
│  • Valida datos          │   │  • Presenta información    │
│  • ORM Django            │   │    al usuario              │
└──────────────────────────┘   └────────────────────────────┘
         │                                │
         │ SQL Queries                    │ HTML Response
         ▼                                ▼
┌──────────────────────────┐         USUARIO
│  DATABASE                │
│  • PostgreSQL (Supabase) │
│  • SQLite (desarrollo)   │
└──────────────────────────┘
```

### Flujo de una Predicción:

```
1. Usuario sube imagen → 2. POST a /predict/ 
   ↓
3. predict_view() recibe request → 4. Valida formulario
   ↓
5. Guarda imagen en modelo → 6. Llama a predictor.py
   ↓
7. Modelo TensorFlow procesa → 8. Retorna probabilidades
   ↓
9. Guarda resultados en DB → 10. Redirige a /results/{id}/
   ↓
11. results_view() lee de DB → 12. Renderiza template
   ↓
13. Usuario ve resultado con gráficas y recomendaciones
```

---

## 📁 Estructura de Archivos

```
django_skin_disease_detector/
│
├── manage.py                      # Script de gestión de Django
├── requirements.txt               # Dependencias Python
├── .env                          # Variables de entorno (credenciales)
├── .env.example                  # Plantilla de variables de entorno
├── db.sqlite3                    # Base de datos SQLite (desarrollo)
│
├── skin_disease_project/         # Configuración del proyecto Django
│   ├── __init__.py
│   ├── settings.py               # ⚙️ Configuración principal
│   ├── urls.py                   # 🔗 URLs principales
│   └── wsgi.py                   # Servidor WSGI para producción
│
├── skin_detector/                # 📦 Aplicación principal
│   ├── __init__.py
│   ├── admin.py                  # 👨‍💼 Panel de administración
│   ├── apps.py                   # Configuración de la app
│   ├── models.py                 # 🗄️ Modelos de datos (DB)
│   ├── views.py                  # 🎮 Lógica de vistas
│   ├── forms.py                  # 📝 Formularios de Django
│   ├── urls.py                   # 🔗 URLs de la app
│   ├── predictor.py              # 🤖 Motor de predicción IA
│   ├── utils.py                  # 🔧 Funciones auxiliares
│   ├── constants.py              # 📊 Constantes (info enfermedades)
│   ├── supabase_utils.py         # ☁️ Utilidades de Supabase
│   └── migrations/               # 📦 Migraciones de base de datos
│
├── templates/                    # 🎨 Plantillas HTML
│   └── skin_detector/
│       ├── base.html             # Plantilla base (navbar, footer)
│       ├── landing.html          # Página de inicio
│       ├── home.html             # Página de diagnóstico
│       ├── prediction_detail.html # Resultados de predicción
│       ├── history.html          # Historial de diagnósticos
│       ├── disease_info.html     # Información de enfermedades
│       ├── register.html         # Registro de usuario
│       └── login.html            # Inicio de sesión
│
├── static/                       # 📦 Archivos estáticos
│   ├── css/
│   │   └── style.css             # Estilos personalizados
│   ├── js/
│   │   └── main.js               # JavaScript personalizado
│   └── images/                   # Imágenes del sitio
│
├── media/                        # 📷 Archivos subidos por usuarios
│   ├── skin_images/              # Imágenes de lesiones
│   └── uploads/                  # Otros archivos
│
├── models/                       # 🧠 Modelos de IA
│   └── improved_balanced_7class_model.h5  # Modelo TensorFlow
│
└── 📚 Documentación/
    ├── README.md                 # Este archivo
    ├── DATABASE.md               # Documentación de Supabase
    ├── AUTH_SYSTEM.md            # Sistema de autenticación
    ├── ARCHITECTURE.md           # Arquitectura del sistema
    └── BEST_PRACTICES.md         # Mejores prácticas
```

---

## 🗄️ Modelos (Models)

Los modelos definen la estructura de datos y se comunican con la base de datos mediante el ORM de Django.

### `SkinImagePrediction` - Modelo Principal

**Ubicación**: `skin_detector/models.py`

**Propósito**: Almacena imágenes de lesiones cutáneas y sus predicciones de IA.

#### Estructura del Modelo:

```python
class SkinImagePrediction(models.Model):
    """
    Representa una predicción individual de enfermedad cutánea.
    """
    
    # ===== DATOS DE LA IMAGEN =====
    image = models.ImageField(
        upload_to='skin_images/',
        validators=[FileExtensionValidator(['jpg', 'jpeg', 'png'])]
    )
    # - Guarda la imagen subida por el usuario
    # - Solo acepta JPG, JPEG, PNG
    # - Se almacena en media/skin_images/
    
    # ===== RESULTADOS DE LA PREDICCIÓN =====
    predicted_class = models.CharField(
        max_length=10,
        choices=DISEASE_CHOICES
    )
    # - Código de la enfermedad predicha (ej: 'mel', 'bcc')
    # - Opciones: akiec, bcc, bkl, df, mel, nv, vasc
    
    confidence_score = models.FloatField()
    # - Nivel de confianza de la predicción (0.0 - 1.0)
    # - Ejemplo: 0.8765 = 87.65% de confianza
    
    probabilities = models.JSONField()
    # - Diccionario con probabilidades de las 7 clases
    # - Ejemplo: {'mel': 0.87, 'nv': 0.08, 'bcc': 0.03, ...}
    
    # ===== METADATOS =====
    uploaded_at = models.DateTimeField(auto_now_add=True)
    # - Fecha y hora de subida automática
    
    processed_at = models.DateTimeField()
    # - Fecha y hora del procesamiento de IA
    
    image_size = models.CharField(max_length=50)
    # - Dimensiones de la imagen (ej: "800x600")
    
    processing_time = models.FloatField()
    # - Tiempo que tardó la predicción en segundos
```

#### Métodos del Modelo:

##### `__str__(self)`
```python
def __str__(self):
    """Representación en string del objeto."""
    return f"Predicción #{self.id} - {self.predicted_class} ({self.uploaded_at})"

# Ejemplo de salida:
# "Predicción #42 - mel (22/10/2025 14:30)"
```

##### `get_predicted_disease_name(self)`
```python
def get_predicted_disease_name(self):
    """
    Obtiene el nombre completo de la enfermedad.
    
    Returns:
        str: Nombre completo en español
    """
    # 'mel' → 'Melanoma'
    # 'bcc' → 'Basal cell carcinoma (Carcinoma basocelular)'
```

##### `get_confidence_percentage(self)`
```python
def get_confidence_percentage(self):
    """
    Convierte confianza a porcentaje.
    
    Returns:
        float: Porcentaje redondeado a 2 decimales
    """
    # 0.8765 → 87.65
    return round(self.confidence_score * 100, 2)
```

##### `is_high_confidence(self, threshold=0.8)`
```python
def is_high_confidence(self, threshold=0.8):
    """
    Verifica si la predicción tiene alta confianza.
    
    Args:
        threshold (float): Umbral de confianza (default 0.8)
    
    Returns:
        bool: True si confianza >= threshold
    """
    # Si confidence_score = 0.85 y threshold = 0.8 → True
    # Si confidence_score = 0.75 y threshold = 0.8 → False
```

##### `get_severity_level(self)`
```python
def get_severity_level(self):
    """
    Determina el nivel de severidad de la enfermedad.
    
    Returns:
        str: 'high', 'medium', 'low', o 'unknown'
    """
    severity_map = {
        'mel': 'high',      # Melanoma - muy peligroso
        'bcc': 'high',      # Carcinoma basocelular - cáncer
        'akiec': 'high',    # Queratosis actínicas - precanceroso
        'vasc': 'medium',   # Lesiones vasculares - seguimiento
        'bkl': 'low',       # Queratosis benigna - no canceroso
        'nv': 'low',        # Nevos - lunares comunes
        'df': 'low'         # Dermatofibroma - benigno
    }
    return severity_map.get(self.predicted_class, 'unknown')
```

#### Uso del Modelo:

```python
# CREAR una nueva predicción
prediction = SkinImagePrediction.objects.create(
    image=uploaded_file,
    predicted_class='mel',
    confidence_score=0.8765,
    probabilities={'mel': 0.8765, 'nv': 0.0823, ...},
    image_size='800x600',
    processing_time=0.342
)

# LEER predicciones
all_predictions = SkinImagePrediction.objects.all()
recent = SkinImagePrediction.objects.order_by('-uploaded_at')[:10]
high_confidence = SkinImagePrediction.objects.filter(confidence_score__gte=0.9)

# ACTUALIZAR una predicción
prediction.predicted_class = 'bcc'
prediction.save()

# ELIMINAR una predicción
prediction.delete()
```

---

## 🎮 Vistas (Views)

Las vistas contienen la lógica de negocio y procesan las peticiones HTTP.

**Ubicación**: `skin_detector/views.py`

### Categorías de Vistas:

1. **Autenticación**: registro, login, logout
2. **Páginas Web**: landing, home, resultados, historial
3. **API REST**: endpoints para aplicaciones móviles

---

### 1️⃣ Vistas de Autenticación

#### `register_view(request)`

**Propósito**: Maneja el registro de nuevos usuarios.

**URL**: `/register/`

**Métodos**: GET, POST

**Qué hace**:

```python
def register_view(request):
    """Vista de registro de nuevos usuarios."""
    
    # Si ya está autenticado, redirigir
    if request.user.is_authenticated:
        return redirect('skin_detector:diagnostico')
    
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        
        if form.is_valid():
            # Crear nuevo usuario
            user = form.save()
            username = form.cleaned_data.get('username')
            
            # Mensaje de éxito
            messages.success(request, f'¡Cuenta creada para {username}!')
            
            # Log del evento
            logger.info(f'Nuevo usuario registrado: {username}')
            
            # Redirigir al login
            return redirect('skin_detector:login')
        else:
            # Mostrar errores del formulario
            messages.error(request, 'Corrige los errores del formulario.')
    
    else:
        # GET: Mostrar formulario vacío
        form = UserRegistrationForm()
    
    # Renderizar template
    return render(request, 'skin_detector/register.html', {
        'form': form,
        'title': 'Registro de Usuario'
    })
```

**Flujo**:
1. Usuario accede a `/register/`
2. Se muestra formulario con 6 campos: username, email, first_name, last_name, password1, password2
3. Usuario completa y envía formulario
4. Django valida:
   - Username único
   - Email válido y único
   - Contraseña segura (min 8 caracteres, no completamente numérica)
   - Contraseñas coinciden
5. Si es válido → crea usuario, muestra mensaje de éxito, redirige a login
6. Si es inválido → muestra errores, mantiene datos ingresados

**Template**: `register.html`

---

#### `login_view(request)`

**Propósito**: Autentica usuarios existentes.

**URL**: `/login/`

**Métodos**: GET, POST

**Qué hace**:

```python
def login_view(request):
    """Vista de inicio de sesión."""
    
    if request.user.is_authenticated:
        return redirect('skin_detector:diagnostico')
    
    if request.method == 'POST':
        form = UserLoginForm(request, data=request.POST)
        
        if form.is_valid():
            # Obtener credenciales
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            
            # Autenticar usuario
            user = authenticate(username=username, password=password)
            
            if user is not None:
                # Crear sesión
                login(request, user)
                
                # Mensaje de bienvenida
                messages.success(request, f'¡Bienvenido, {username}!')
                
                # Log del evento
                logger.info(f'Usuario {username} inició sesión')
                
                # Redirigir a página solicitada o diagnóstico
                next_page = request.GET.get('next', 'skin_detector:diagnostico')
                return redirect(next_page)
            else:
                messages.error(request, 'Usuario o contraseña incorrectos.')
        else:
            messages.error(request, 'Datos inválidos.')
    
    else:
        form = UserLoginForm()
    
    return render(request, 'skin_detector/login.html', {
        'form': form,
        'title': 'Iniciar Sesión'
    })
```

**Flujo**:
1. Usuario accede a `/login/`
2. Ingresa username y password
3. Django autentica credenciales con bcrypt
4. Si son correctas:
   - Crea sesión en servidor
   - Guarda cookie en navegador
   - Redirige a página de diagnóstico
5. Si son incorrectas:
   - Muestra mensaje de error
   - No revela si username o password son incorrectos (seguridad)

**Características**:
- Checkbox "Recuérdame" (mantiene sesión activa)
- Soporte para parámetro `?next=/url/` (redirige después de login)
- Protección contra ataques de fuerza bruta (Django lo maneja automáticamente)

---

#### `logout_view(request)`

**Propósito**: Cierra la sesión del usuario.

**URL**: `/logout/`

**Métodos**: POST

**Decorator**: `@login_required`

**Qué hace**:

```python
@login_required
def logout_view(request):
    """Vista de cierre de sesión."""
    
    username = request.user.username
    
    # Destruir sesión
    logout(request)
    
    # Mensaje de despedida
    messages.info(request, f'Has cerrado sesión, {username}. ¡Hasta pronto!')
    
    # Log del evento
    logger.info(f'Usuario {username} cerró sesión')
    
    # Redirigir a landing
    return redirect('skin_detector:landing')
```

**Flujo**:
1. Usuario hace clic en "Cerrar Sesión" (dropdown en navbar)
2. Se envía POST a `/logout/`
3. Django destruye la sesión en servidor
4. Elimina cookie del navegador
5. Redirige a página de inicio

**Seguridad**:
- Solo acepta método POST (previene CSRF)
- Requiere autenticación previa (`@login_required`)

---

### 2️⃣ Vistas de Páginas Web

#### `landing_view(request)`

**Propósito**: Página de inicio del sitio.

**URL**: `/` o `/landing/`

**Qué hace**:
```python
def landing_view(request):
    """Página de bienvenida con información del sistema."""
    return render(request, 'skin_detector/landing.html', {
        'title': 'DermatologIA - Detección de Enfermedades Cutáneas',
        'stats': {
            'total_predictions': SkinImagePrediction.objects.count(),
            'accuracy': '85%',
            'diseases_detected': 7
        }
    })
```

**Contenido**:
- Hero section con llamado a la acción
- Características del sistema
- Estadísticas generales
- Botones para registro/login o diagnóstico

---

#### `home_view(request)` / `diagnostico_view(request)`

**Propósito**: Página principal de diagnóstico donde se sube la imagen.

**URL**: `/diagnostico/` o `/home/`

**Decorator**: `@login_required` (opcional, depende de configuración)

**Qué hace**:

```python
def diagnostico_view(request):
    """
    Página de diagnóstico con formulario de subida de imagen.
    
    Muestra dos opciones:
    1. Subir imagen desde computadora
    2. Tomar foto con cámara (en móviles)
    """
    
    if request.method == 'POST':
        # Procesar subida de imagen
        form = SkinImageUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            # Guardar imagen temporalmente
            prediction = form.save(commit=False)
            
            # Si hay usuario autenticado, asociarlo
            if request.user.is_authenticated:
                prediction.user = request.user
            
            prediction.save()
            
            # Redirigir a vista de procesamiento
            return redirect('skin_detector:predict', pk=prediction.id)
        else:
            messages.error(request, 'Error al subir la imagen.')
    
    else:
        form = SkinImageUploadForm()
    
    # Estadísticas del usuario (si está autenticado)
    user_stats = None
    if request.user.is_authenticated:
        user_predictions = SkinImagePrediction.objects.filter(user=request.user)
        user_stats = {
            'total': user_predictions.count(),
            'high_risk': user_predictions.filter(
                predicted_class__in=['mel', 'bcc', 'akiec']
            ).count(),
            'recent': user_predictions.order_by('-uploaded_at')[:5]
        }
    
    return render(request, 'skin_detector/home.html', {
        'form': form,
        'user_stats': user_stats,
        'title': 'Diagnóstico de Lesiones Cutáneas'
    })
```

**Elementos del formulario**:
- Input de archivo con drag & drop
- Vista previa de imagen
- Botón de cámara (en móviles)
- Validación de tamaño y formato

---

#### `predict_view(request, pk)`

**Propósito**: Procesa la imagen con el modelo de IA.

**URL**: `/predict/<int:pk>/`

**Qué hace**:

```python
def predict_view(request, pk):
    """
    Procesa la imagen con el modelo de IA y guarda resultados.
    
    Args:
        pk (int): ID de la predicción (SkinImagePrediction.id)
    """
    
    # Obtener el objeto de predicción
    prediction = get_object_or_404(SkinImagePrediction, pk=pk)
    
    # Verificar que aún no se haya procesado
    if prediction.processed_at is not None:
        return redirect('skin_detector:prediction_detail', pk=pk)
    
    try:
        # Obtener instancia del predictor (singleton)
        predictor = get_predictor()
        
        # Registrar inicio de procesamiento
        start_time = time.time()
        prediction.processed_at = timezone.now()
        
        # Obtener ruta completa de la imagen
        image_path = prediction.image.path
        
        # EJECUTAR PREDICCIÓN
        result = predictor.predict(image_path)
        
        # Calcular tiempo de procesamiento
        processing_time = time.time() - start_time
        
        # Guardar resultados en el modelo
        prediction.predicted_class = result['predicted_class']
        prediction.confidence_score = result['confidence']
        prediction.probabilities = result['probabilities']
        prediction.processing_time = processing_time
        
        # Obtener dimensiones de la imagen
        from PIL import Image
        img = Image.open(image_path)
        prediction.image_size = f"{img.width}x{img.height}"
        
        # Guardar en la base de datos
        prediction.save()
        
        # Mensaje de éxito
        messages.success(
            request,
            f'Análisis completado: {prediction.get_predicted_disease_name()} '
            f'con {prediction.get_confidence_percentage()}% de confianza.'
        )
        
        # Log del evento
        logger.info(
            f'Predicción #{pk} procesada: {prediction.predicted_class} '
            f'({prediction.confidence_score:.4f}) en {processing_time:.2f}s'
        )
        
        # Redirigir a resultados
        return redirect('skin_detector:prediction_detail', pk=pk)
    
    except Exception as e:
        # Manejo de errores
        logger.error(f'Error en predicción #{pk}: {str(e)}')
        messages.error(
            request,
            f'Error al procesar la imagen: {str(e)}'
        )
        return redirect('skin_detector:diagnostico')
```

**Flujo**:
1. Usuario sube imagen → se crea registro en DB
2. Vista `predict` carga el modelo de TensorFlow
3. Preprocesa imagen (resize 224x224, normalización)
4. Ejecuta inferencia del modelo
5. Obtiene probabilidades de las 7 clases
6. Guarda resultados en DB
7. Redirige a vista de resultados

**Tiempo de procesamiento típico**: 0.3-1.5 segundos

---

#### `prediction_detail_view(request, pk)`

**Propósito**: Muestra los resultados de la predicción.

**URL**: `/results/<int:pk>/`

**Qué hace**:

```python
def prediction_detail_view(request, pk):
    """
    Muestra los resultados detallados de una predicción.
    
    Args:
        pk (int): ID de la predicción
    """
    
    # Obtener predicción o 404
    prediction = get_object_or_404(SkinImagePrediction, pk=pk)
    
    # Verificar que ya esté procesada
    if not prediction.processed_at:
        messages.warning(request, 'Esta predicción aún no ha sido procesada.')
        return redirect('skin_detector:predict', pk=pk)
    
    # Preparar datos para la gráfica de probabilidades
    probabilities_sorted = sorted(
        prediction.probabilities.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Obtener información de la enfermedad desde constants.py
    from .constants import DISEASE_INFO
    disease_info = DISEASE_INFO.get(prediction.predicted_class, {})
    
    # Determinar nivel de riesgo
    severity = prediction.get_severity_level()
    risk_color = {
        'high': 'danger',
        'medium': 'warning',
        'low': 'success'
    }.get(severity, 'info')
    
    # Recomendaciones personalizadas
    recommendations = []
    
    if severity == 'high':
        recommendations = [
            '⚠️ Consulta con un dermatólogo lo antes posible',
            '📅 Programa una cita médica en menos de 1 semana',
            '📝 Lleva esta imagen a tu consulta',
            '🔍 No esperes a que los síntomas empeoren'
        ]
    elif severity == 'medium':
        recommendations = [
            '👨‍⚕️ Consulta con un dermatólogo en las próximas semanas',
            '📸 Toma fotos periódicas para monitorear cambios',
            '📋 Anota cualquier síntoma nuevo'
        ]
    else:  # low
        recommendations = [
            '✅ Esta lesión parece benigna',
            '👀 Monitorea cualquier cambio en tamaño, forma o color',
            '📅 Revisión anual con dermatólogo recomendada'
        ]
    
    # Buscar predicciones similares (opcional)
    similar_predictions = SkinImagePrediction.objects.filter(
        predicted_class=prediction.predicted_class
    ).exclude(pk=pk).order_by('-confidence_score')[:5]
    
    context = {
        'prediction': prediction,
        'probabilities_sorted': probabilities_sorted,
        'disease_info': disease_info,
        'severity': severity,
        'risk_color': risk_color,
        'recommendations': recommendations,
        'similar_predictions': similar_predictions,
        'title': f'Resultado: {prediction.get_predicted_disease_name()}'
    }
    
    return render(request, 'skin_detector/prediction_detail.html', context)
```

**Contenido de la página**:
- Imagen subida con zoom
- Resultado principal con badge de confianza
- Gráfica de barras con probabilidades de las 7 clases
- Información detallada de la enfermedad:
  - Descripción
  - Síntomas
  - Causas
  - Tratamientos
  - Prevención
- Nivel de riesgo (alto/medio/bajo)
- Recomendaciones personalizadas
- Botón para descargar reporte PDF (opcional)
- Botones de compartir (opcional)

---

#### `history_view(request)`

**Propósito**: Muestra el historial de predicciones del usuario.

**URL**: `/history/`

**Decorator**: `@login_required`

**Qué hace**:

```python
@login_required
def history_view(request):
    """
    Muestra el historial de predicciones del usuario autenticado.
    """
    
    # Obtener todas las predicciones del usuario
    predictions = SkinImagePrediction.objects.filter(
        user=request.user
    ).order_by('-uploaded_at')
    
    # Paginación (10 por página)
    from django.core.paginator import Paginator
    paginator = Paginator(predictions, 10)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)
    
    # Estadísticas del usuario
    stats = {
        'total': predictions.count(),
        'high_risk': predictions.filter(
            predicted_class__in=['mel', 'bcc', 'akiec']
        ).count(),
        'average_confidence': predictions.aggregate(
            avg_conf=models.Avg('confidence_score')
        )['avg_conf'] or 0,
        'most_common': predictions.values('predicted_class').annotate(
            count=models.Count('id')
        ).order_by('-count').first()
    }
    
    context = {
        'page_obj': page_obj,
        'stats': stats,
        'title': 'Historial de Diagnósticos'
    }
    
    return render(request, 'skin_detector/history.html', context)
```

**Elementos de la página**:
- Tabla con todas las predicciones:
  - Fecha
  - Miniatura de imagen
  - Enfermedad detectada
  - Nivel de confianza
  - Botón "Ver detalles"
  - Botón "Eliminar"
- Estadísticas del usuario:
  - Total de diagnósticos
  - Diagnósticos de alto riesgo
  - Confianza promedio
  - Enfermedad más común
- Paginación (10 resultados por página)
- Filtros (por fecha, enfermedad, confianza)

---

#### `disease_info_view(request, disease_code)`

**Propósito**: Muestra información detallada de una enfermedad específica.

**URL**: `/diseases/<str:disease_code>/`

**Qué hace**:

```python
def disease_info_view(request, disease_code):
    """
    Muestra información completa de una enfermedad.
    
    Args:
        disease_code (str): Código de la enfermedad (akiec, bcc, bkl, df, mel, nv, vasc)
    """
    
    from .constants import DISEASE_INFO
    
    # Verificar que el código sea válido
    if disease_code not in DISEASE_INFO:
        messages.error(request, 'Enfermedad no encontrada.')
        return redirect('skin_detector:landing')
    
    # Obtener información de la enfermedad
    disease_info = DISEASE_INFO[disease_code]
    
    # Estadísticas de esta enfermedad en la base de datos
    stats = {
        'total_cases': SkinImagePrediction.objects.filter(
            predicted_class=disease_code
        ).count(),
        'average_confidence': SkinImagePrediction.objects.filter(
            predicted_class=disease_code
        ).aggregate(avg=models.Avg('confidence_score'))['avg'] or 0
    }
    
    context = {
        'disease_code': disease_code,
        'disease_info': disease_info,
        'stats': stats,
        'title': disease_info['name']
    }
    
    return render(request, 'skin_detector/disease_info.html', context)
```

**Contenido**:
- Nombre completo de la enfermedad
- Descripción detallada
- Imágenes de referencia
- Síntomas y signos
- Factores de riesgo
- Diagnóstico médico
- Opciones de tratamiento
- Pronóstico
- Prevención
- Referencias médicas

---

### 3️⃣ API REST

El sistema incluye endpoints API para integración con aplicaciones móviles.

#### `api_predict_view(request)`

**Propósito**: Endpoint para predicciones desde apps móviles.

**URL**: `/api/predict/`

**Método**: POST

**Decorators**: `@api_view(['POST'])`, `@csrf_exempt`

**Qué hace**:

```python
@api_view(['POST'])
@permission_classes([AllowAny])
def api_predict_view(request):
    """
    API endpoint para predicción de enfermedades cutáneas.
    
    Request:
        POST /api/predict/
        Content-Type: multipart/form-data
        Body: {
            "image": <archivo de imagen>
        }
    
    Response:
        {
            "success": true,
            "prediction_id": 42,
            "predicted_class": "mel",
            "predicted_disease": "Melanoma",
            "confidence": 0.8765,
            "confidence_percentage": 87.65,
            "probabilities": {
                "mel": 0.8765,
                "nv": 0.0823,
                "bcc": 0.0287,
                ...
            },
            "severity": "high",
            "processing_time": 0.342,
            "image_url": "/media/skin_images/image_42.jpg"
        }
    """
    
    # Validar que se envió una imagen
    if 'image' not in request.FILES:
        return Response({
            'success': False,
            'error': 'No se proporcionó ninguna imagen'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        # Crear objeto de predicción
        prediction = SkinImagePrediction.objects.create(
            image=request.FILES['image']
        )
        
        # Procesar con IA
        predictor = get_predictor()
        start_time = time.time()
        
        result = predictor.predict(prediction.image.path)
        processing_time = time.time() - start_time
        
        # Actualizar modelo
        prediction.predicted_class = result['predicted_class']
        prediction.confidence_score = result['confidence']
        prediction.probabilities = result['probabilities']
        prediction.processing_time = processing_time
        prediction.processed_at = timezone.now()
        prediction.save()
        
        # Construir respuesta
        response_data = {
            'success': True,
            'prediction_id': prediction.id,
            'predicted_class': prediction.predicted_class,
            'predicted_disease': prediction.get_predicted_disease_name(),
            'confidence': prediction.confidence_score,
            'confidence_percentage': prediction.get_confidence_percentage(),
            'probabilities': prediction.probabilities,
            'severity': prediction.get_severity_level(),
            'processing_time': processing_time,
            'image_url': request.build_absolute_uri(prediction.image.url),
            'uploaded_at': prediction.uploaded_at.isoformat(),
            'processed_at': prediction.processed_at.isoformat()
        }
        
        return Response(response_data, status=status.HTTP_200_OK)
    
    except Exception as e:
        logger.error(f'Error en API predict: {str(e)}')
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
```

**Ejemplo de uso (curl)**:
```bash
curl -X POST http://localhost:8000/api/predict/ \
  -H "Content-Type: multipart/form-data" \
  -F "image=@lesion.jpg"
```

**Ejemplo de uso (Python requests)**:
```python
import requests

url = 'http://localhost:8000/api/predict/'
files = {'image': open('lesion.jpg', 'rb')}

response = requests.post(url, files=files)
result = response.json()

print(f"Predicción: {result['predicted_disease']}")
print(f"Confianza: {result['confidence_percentage']}%")
```

---

#### `api_history_view(request)`

**Propósito**: Endpoint para obtener historial de predicciones.

**URL**: `/api/history/`

**Método**: GET

**Qué hace**:

```python
@api_view(['GET'])
@permission_classes([AllowAny])
def api_history_view(request):
    """
    API endpoint para obtener historial de predicciones.
    
    Query Parameters:
        - limit (int): Número máximo de resultados (default: 20)
        - offset (int): Offset para paginación (default: 0)
        - disease (str): Filtrar por código de enfermedad
        - min_confidence (float): Confianza mínima (0.0-1.0)
    
    Response:
        {
            "count": 150,
            "results": [
                {
                    "id": 42,
                    "predicted_class": "mel",
                    "predicted_disease": "Melanoma",
                    "confidence": 0.8765,
                    "uploaded_at": "2025-10-22T14:30:00Z",
                    "image_url": "/media/skin_images/..."
                },
                ...
            ]
        }
    """
    
    # Obtener parámetros de consulta
    limit = int(request.GET.get('limit', 20))
    offset = int(request.GET.get('offset', 0))
    disease = request.GET.get('disease', None)
    min_confidence = float(request.GET.get('min_confidence', 0.0))
    
    # Consultar base de datos
    queryset = SkinImagePrediction.objects.all()
    
    # Aplicar filtros
    if disease:
        queryset = queryset.filter(predicted_class=disease)
    if min_confidence > 0:
        queryset = queryset.filter(confidence_score__gte=min_confidence)
    
    # Ordenar y paginar
    queryset = queryset.order_by('-uploaded_at')[offset:offset+limit]
    
    # Serializar resultados
    results = []
    for pred in queryset:
        results.append({
            'id': pred.id,
            'predicted_class': pred.predicted_class,
            'predicted_disease': pred.get_predicted_disease_name(),
            'confidence': pred.confidence_score,
            'confidence_percentage': pred.get_confidence_percentage(),
            'uploaded_at': pred.uploaded_at.isoformat(),
            'image_url': request.build_absolute_uri(pred.image.url)
        })
    
    return Response({
        'count': SkinImagePrediction.objects.count(),
        'results': results
    }, status=status.HTTP_200_OK)
```

---

## 📝 Formularios (Forms)

**Ubicación**: `skin_detector/forms.py`

Los formularios de Django manejan validación de datos del usuario.

### `SkinImageUploadForm`

```python
from django import forms
from .models import SkinImagePrediction

class SkinImageUploadForm(forms.ModelForm):
    """
    Formulario para subir imágenes de lesiones cutáneas.
    """
    
    class Meta:
        model = SkinImagePrediction
        fields = ['image']
        widgets = {
            'image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/jpeg,image/jpg,image/png',
                'id': 'imageInput'
            })
        }
    
    def clean_image(self):
        """
        Valida que la imagen cumpla con los requisitos.
        """
        image = self.cleaned_data.get('image')
        
        if image:
            # Validar tamaño (máx 5MB)
            if image.size > 5 * 1024 * 1024:
                raise forms.ValidationError(
                    'La imagen es demasiado grande (máx 5MB).'
                )
            
            # Validar formato
            valid_extensions = ['.jpg', '.jpeg', '.png']
            ext = os.path.splitext(image.name)[1].lower()
            if ext not in valid_extensions:
                raise forms.ValidationError(
                    'Formato no válido. Usa JPG, JPEG o PNG.'
                )
        
        return image
```

### `UserRegistrationForm`

```python
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class UserRegistrationForm(UserCreationForm):
    """
    Formulario de registro con campos adicionales.
    """
    
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'correo@ejemplo.com'
        })
    )
    
    first_name = forms.CharField(
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Nombre'
        })
    )
    
    last_name = forms.CharField(
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Apellido'
        })
    )
    
    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name', 'password1', 'password2']
    
    def clean_email(self):
        """Valida que el email sea único."""
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError('Este email ya está registrado.')
        return email
    
    def save(self, commit=True):
        """Guarda el usuario con los campos adicionales."""
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        
        if commit:
            user.save()
        
        return user
```

### `UserLoginForm`

```python
from django.contrib.auth.forms import AuthenticationForm

class UserLoginForm(AuthenticationForm):
    """
    Formulario de login personalizado.
    """
    
    username = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Usuario',
            'autofocus': True
        })
    )
    
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Contraseña'
        })
    )
    
    remember_me = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input'
        })
    )
```

---

## 🔗 URLs y Rutas

**Ubicación**: `skin_detector/urls.py`

```python
from django.urls import path
from . import views

app_name = 'skin_detector'

urlpatterns = [
    # Autenticación
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    
    # Páginas Web
    path('', views.landing_view, name='landing'),
    path('diagnostico/', views.diagnostico_view, name='diagnostico'),
    path('predict/<int:pk>/', views.predict_view, name='predict'),
    path('results/<int:pk>/', views.prediction_detail_view, name='prediction_detail'),
    path('history/', views.history_view, name='history'),
    path('diseases/<str:disease_code>/', views.disease_info_view, name='disease_info'),
    
    # API REST
    path('api/predict/', views.api_predict_view, name='api_predict'),
    path('api/history/', views.api_history_view, name='api_history'),
]
```

**URLs principales** (`skin_disease_project/urls.py`):

```python
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('skin_detector.urls')),
]

# Servir archivos media en desarrollo
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

---

## 🎨 Templates

Los templates usan Django Template Language (DTL) con Bootstrap 5.

### Estructura de Template Base:

```django
{% load static %}
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}DermatologIA{% endblock %}</title>
    
    <!-- Bootstrap 5 -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Custom CSS -->
    <link rel="stylesheet" href="{% static 'css/style.css' %}">
    
    {% block extra_css %}{% endblock %}
</head>
<body>
    <!-- Navbar -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="{% url 'skin_detector:landing' %}">
                <i class="fas fa-heartbeat"></i> DermatologIA
            </a>
            
            <div class="collapse navbar-collapse">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="{% url 'skin_detector:diagnostico' %}">
                            <i class="fas fa-microscope"></i> Diagnóstico
                        </a>
                    </li>
                    
                    {% if user.is_authenticated %}
                        <li class="nav-item dropdown">
                            <a class="nav-link dropdown-toggle" href="#" id="userDropdown" 
                               data-bs-toggle="dropdown">
                                <i class="fas fa-user"></i> {{ user.username }}
                            </a>
                            <ul class="dropdown-menu dropdown-menu-end">
                                <li>
                                    <a class="dropdown-item" href="{% url 'skin_detector:history' %}">
                                        <i class="fas fa-history"></i> Mis Diagnósticos
                                    </a>
                                </li>
                                <li><hr class="dropdown-divider"></li>
                                <li>
                                    <form method="POST" action="{% url 'skin_detector:logout' %}">
                                        {% csrf_token %}
                                        <button type="submit" class="dropdown-item">
                                            <i class="fas fa-sign-out-alt"></i> Cerrar Sesión
                                        </button>
                                    </form>
                                </li>
                            </ul>
                        </li>
                    {% else %}
                        <li class="nav-item">
                            <a class="nav-link" href="{% url 'skin_detector:login' %}">
                                <i class="fas fa-sign-in-alt"></i> Iniciar Sesión
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="btn btn-outline-light" href="{% url 'skin_detector:register' %}">
                                Registrarse
                            </a>
                        </li>
                    {% endif %}
                </ul>
            </div>
        </div>
    </nav>
    
    <!-- Messages -->
    {% if messages %}
        <div class="container mt-3">
            {% for message in messages %}
                <div class="alert alert-{{ message.tags }} alert-dismissible fade show">
                    {{ message }}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            {% endfor %}
        </div>
    {% endif %}
    
    <!-- Main Content -->
    <main class="container my-5">
        {% block content %}{% endblock %}
    </main>
    
    <!-- Footer -->
    <footer class="bg-dark text-white text-center py-4 mt-5">
        <p>&copy; 2025 DermatologIA. Todos los derechos reservados.</p>
        <p>
            <small>⚠️ Esta herramienta es solo para propósitos educativos. 
            Consulta con un dermatólogo profesional.</small>
        </p>
    </footer>
    
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <!-- Custom JS -->
    <script src="{% static 'js/main.js' %}"></script>
    
    {% block extra_js %}{% endblock %}
</body>
</html>
```

---

## 🔐 Sistema de Autenticación

### Características:

✅ **Registro de usuarios**: Formulario con validación de email único
✅ **Inicio de sesión**: Autenticación con contraseña hasheada
✅ **Cerrar sesión**: Destrucción segura de sesión
✅ **Protección de rutas**: Decorador `@login_required`
✅ **Mensajes flash**: Feedback visual para usuarios
✅ **Seguridad**: CSRF protection, password hashing con PBKDF2

### Flujo de Autenticación:

```
REGISTRO:
1. Usuario llena formulario → 2. Django valida datos
   ↓
3. Hash de contraseña con PBKDF2 → 4. Guarda en DB
   ↓
5. Redirige a login → 6. Usuario inicia sesión

LOGIN:
1. Usuario ingresa credenciales → 2. Django busca username en DB
   ↓
3. Compara hash de contraseña → 4. Si coincide: crea sesión
   ↓
5. Guarda session_key en cookie → 6. Redirige a diagnóstico

LOGOUT:
1. Usuario hace clic en cerrar sesión → 2. Django invalida session_key
   ↓
3. Elimina cookie del navegador → 4. Redirige a landing
```

---

## 🌐 API REST

### Endpoints Disponibles:

| Método | Endpoint | Descripción | Autenticación |
|--------|----------|-------------|---------------|
| POST | `/api/predict/` | Predecir enfermedad desde imagen | No requerida |
| GET | `/api/history/` | Obtener historial de predicciones | No requerida |
| GET | `/api/diseases/` | Listar información de enfermedades | No requerida |

### Ejemplo de Integración (Flutter/React Native):

```dart
// Flutter ejemplo
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';

Future<Map<String, dynamic>> predictDisease(File imageFile) async {
  var uri = Uri.parse('http://tu-servidor.com/api/predict/');
  var request = http.MultipartRequest('POST', uri);
  
  // Agregar imagen
  request.files.add(
    await http.MultipartFile.fromPath('image', imageFile.path)
  );
  
  // Enviar request
  var response = await request.send();
  var responseData = await response.stream.bytesToString();
  var jsonResponse = json.decode(responseData);
  
  if (jsonResponse['success']) {
    return jsonResponse;
  } else {
    throw Exception(jsonResponse['error']);
  }
}

// Uso:
File image = File('path/to/image.jpg');
var result = await predictDisease(image);

print('Enfermedad: ${result['predicted_disease']}');
print('Confianza: ${result['confidence_percentage']}%');
```

---

## 💾 Base de Datos

### SQLite (Desarrollo):
- Archivo: `db.sqlite3`
- Ubicación: Raíz del proyecto
- Ligera y portable

### PostgreSQL (Producción - Supabase):
- Host: `cpjmodytpeuybpcayzwk.supabase.co`
- Puerto: 5432
- Base de datos: `postgres`
- Configuración en `.env`

### Tablas Principales:

1. **auth_user**: Usuarios del sistema (Django built-in)
2. **skin_detector_skinimageprediction**: Predicciones
3. **django_session**: Sesiones activas
4. **django_migrations**: Historial de migraciones

### Migraciones:

```bash
# Crear migraciones
python manage.py makemigrations

# Aplicar migraciones
python manage.py migrate

# Ver migraciones aplicadas
python manage.py showmigrations
```

---

## ⚙️ Instalación y Configuración

### 1. Clonar repositorio:
```bash
git clone https://github.com/tu-usuario/skin-disease-detector.git
cd skin-disease-detector/django_skin_disease_detector
```

### 2. Crear entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno:
```bash
# Copiar template
cp .env.example .env

# Editar .env con tus credenciales
SECRET_KEY=tu-secret-key-generada
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
DB_PASSWORD=tu-password
SUPABASE_URL=tu-supabase-url
SUPABASE_ANON_KEY=tu-supabase-key
```

### 5. Ejecutar migraciones:
```bash
python manage.py migrate
```

### 6. Crear superusuario (admin):
```bash
python manage.py createsuperuser
```

### 7. Ejecutar servidor:
```bash
python manage.py runserver
```

### 8. Acceder:
- Web: http://127.0.0.1:8000/
- Admin: http://127.0.0.1:8000/admin/

---

## 🚀 Uso del Sistema

### Para Usuarios:

1. **Registrarse**: `/register/`
2. **Iniciar sesión**: `/login/`
3. **Subir imagen**: `/diagnostico/`
4. **Ver resultado**: `/results/{id}/`
5. **Ver historial**: `/history/`

### Para Desarrolladores:

```python
# Usar el predictor manualmente
from skin_detector.predictor import get_predictor

predictor = get_predictor()
result = predictor.predict('path/to/image.jpg')

print(result)
# {
#     'predicted_class': 'mel',
#     'confidence': 0.8765,
#     'probabilities': {'mel': 0.8765, 'nv': 0.0823, ...}
# }
```

---

## 💻 Tecnologías Utilizadas

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| **Python** | 3.11+ | Lenguaje principal |
| **Django** | 4.2.7 | Framework web |
| **TensorFlow** | 2.15.0 | Modelo de IA |
| **Keras** | 2.15.0 | API de alto nivel |
| **Pillow** | 10.0.0+ | Procesamiento de imágenes |
| **NumPy** | 1.24.0+ | Operaciones numéricas |
| **PostgreSQL** | 15+ | Base de datos producción |
| **Supabase** | 2.3.4 | BaaS (Backend as a Service) |
| **Bootstrap** | 5.3 | Framework CSS |
| **Font Awesome** | 6.4 | Iconos |
| **Django REST Framework** | 3.14.0 | API REST |

---

## 📚 Documentación Adicional

- [DATABASE.md](DATABASE.md): Documentación de Supabase
- [AUTH_SYSTEM.md](AUTH_SYSTEM.md): Sistema de autenticación
- [ARCHITECTURE.md](ARCHITECTURE.md): Arquitectura completa
- [BEST_PRACTICES.md](BEST_PRACTICES.md): Mejores prácticas

---

## 🐛 Solución de Problemas

### Error: "No module named 'tensorflow'"
```bash
pip install tensorflow==2.15.0
```

### Error: "Model file not found"
```bash
# Asegúrate de que existe: models/improved_balanced_7class_model.h5
ls models/
```

### Error: "CSRF token missing"
```python
# En templates, agrega {% csrf_token %} en formularios POST
<form method="POST">
    {% csrf_token %}
    ...
</form>
```

---

## 📄 Licencia

Este proyecto es para propósitos educativos.

⚠️ **Advertencia médica**: No usar para diagnósticos médicos reales sin supervisión profesional.

---

**¡Sistema completo y funcional! 🎉**

Para más información, consulta la documentación adicional o contacta al equipo de desarrollo.
