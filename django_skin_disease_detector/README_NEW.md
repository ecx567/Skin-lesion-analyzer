# 🏥 SkinAI - Sistema de Detección de Enfermedades Cutáneas

<div align="center">

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Django](https://img.shields.io/badge/Django-5.2.7-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange)
![License](https://img.shields.io/badge/license-MIT-purple)

**Sistema inteligente de detección de enfermedades cutáneas mediante Deep Learning**

[🚀 Demo](#demo) • [📖 Documentación](#documentación) • [🛠️ Instalación](#instalación) • [📝 Uso](#uso) • [🤝 Contribuir](#contribuir)

</div>

---

## 📋 Tabla de Contenidos

- [🎯 Descripción](#-descripción)
- [✨ Características](#-características)
- [🧠 Modelo de IA](#-modelo-de-ia)
- [🛠️ Instalación](#️-instalación)
  - [Requisitos Previos](#requisitos-previos)
  - [Instalación Local](#instalación-local)
  - [Variables de Entorno](#variables-de-entorno)
- [📝 Guía de Uso](#-guía-de-uso)
  - [Para Usuarios](#para-usuarios)
  - [Para Desarrolladores](#para-desarrolladores)
- [📚 Guía Técnica](#-guía-técnica)
  - [Arquitectura](#arquitectura)
  - [Estructura del Proyecto](#estructura-del-proyecto)
  - [API REST](#api-rest)
- [🔧 Configuración Avanzada](#-configuración-avanzada)
- [🧪 Testing](#-testing)
- [🚀 Despliegue](#-despliegue)
- [🤝 Contribuir](#-contribuir)
- [📄 Licencia](#-licencia)

---

## 🎯 Descripción

**SkinAI** es una aplicación web de diagnóstico asistido por inteligencia artificial para la detección temprana de enfermedades cutáneas. Utiliza un modelo de Deep Learning entrenado con el dataset HAM10000 para clasificar lesiones de piel en 7 categorías diferentes.

### 🎓 Dataset HAM10000

El modelo está entrenado con el reconocido dataset **HAM10000** (Human Against Machine with 10000 training images), que incluye:
- **10,015 imágenes dermoscópicas**
- **7 tipos de lesiones cutáneas**
- **Validado por dermatólogos certificados**

### 🏆 Tipos de Lesiones Detectadas

| Código | Nombre | Descripción | Prevalencia |
|--------|--------|-------------|-------------|
| **MEL** | Melanoma | Cáncer de piel maligno | Alta prioridad |
| **NV** | Nevus Melanocítico | Lunar común benigno | Más común |
| **BCC** | Carcinoma Basocelular | Cáncer de piel no melanoma | Común |
| **AKIEC** | Queratosis Actínica | Lesión precancerosa | Media |
| **BKL** | Queratosis Benigna | Lesión benigna relacionada con la edad | Común |
| **DF** | Dermatofibroma | Tumor benigno de la piel | Poco común |
| **VASC** | Lesión Vascular | Anomalías de vasos sanguíneos | Poco común |

---

## ✨ Características

### 🔬 Funcionalidades Principales

- ✅ **Clasificación de 7 tipos** de lesiones cutáneas con >85% de precisión
- ✅ **Subida de imágenes** desde dispositivo o cámara
- ✅ **Análisis en tiempo real** con resultados instantáneos
- ✅ **Historial de diagnósticos** personalizado por usuario
- ✅ **Información detallada** sobre cada enfermedad detectada
- ✅ **Reportes en PDF** descargables con resultados
- ✅ **Envío por email** de reportes de diagnóstico
- ✅ **Visualización de confianza** con gráficos interactivos

### 👥 Sistema de Usuarios

- 🔐 **Autenticación completa**: Registro, login, recuperación de contraseña
- 🔑 **OAuth2 con Google**: Inicio de sesión con cuenta de Google
- 👤 **Perfiles de usuario**: Gestión de información personal
- 📊 **Dashboard personalizado**: Historial y estadísticas

### 🌐 API REST

- 📡 **Endpoints RESTful** para integración con apps móviles
- 🔒 **Autenticación JWT** para acceso seguro
- 📝 **Documentación Swagger** automática
- 🔄 **Respuestas JSON** estandarizadas

### ☁️ Infraestructura

- 🗄️ **Base de datos**: SQLite (desarrollo) / PostgreSQL-Supabase (producción)
- 📦 **Almacenamiento**: Media files en servidor / Supabase Storage
- 🚀 **Escalable**: Diseño modular y preparado para producción
- 🔒 **Seguro**: HTTPS, sanitización de inputs, protección CSRF

---

## 🧠 Modelo de IA

### Arquitectura del Modelo

```
Input: Imagen RGB (224x224x3)
    ↓
[Convolutional Layers con Data Augmentation]
    ↓
[Batch Normalization + Dropout (0.3)]
    ↓
[Dense Layers (128 → 64 neuronas)]
    ↓
Output: 7 clases (Softmax)
```

### Métricas de Rendimiento

| Métrica | Valor | Descripción |
|---------|-------|-------------|
| **Accuracy** | 85.3% | Precisión global del modelo |
| **Precision** | 83.7% | Predicciones correctas sobre el total predicho |
| **Recall** | 84.1% | Casos correctamente identificados |
| **F1-Score** | 83.9% | Balance entre precisión y recall |

### Técnicas Utilizadas

- ✅ **Transfer Learning** con arquitectura base optimizada
- ✅ **Data Augmentation** (rotación, flip, zoom, brillo)
- ✅ **Class Weighting** para balance de clases
- ✅ **Early Stopping** para evitar overfitting
- ✅ **Dropout Layers** para mejor generalización

---

## 🛠️ Instalación

### Requisitos Previos

Asegúrate de tener instalado:

- **Python 3.11+** ([Descargar](https://www.python.org/downloads/))
- **Git** ([Descargar](https://git-scm.com/downloads))
- **pip** (incluido con Python)
- **virtualenv** (recomendado)

```bash
# Verificar instalaciones
python --version  # Python 3.11.x
pip --version     # pip 23.x
git --version     # git 2.x
```

### Instalación Local

#### 1️⃣ Clonar el Repositorio

```bash
# Clonar desde GitHub
git clone https://github.com/ecx567/Skin-lesion-analyzer.git
cd Skin-lesion-analyzer/django_skin_disease_detector
```

#### 2️⃣ Crear Entorno Virtual

**Windows (PowerShell/CMD):**
```powershell
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
.\venv\Scripts\activate  # PowerShell
# o
venv\Scripts\activate.bat  # CMD
```

**Linux/Mac:**
```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
source venv/bin/activate
```

#### 3️⃣ Instalar Dependencias

```bash
# Actualizar pip
python -m pip install --upgrade pip

# Instalar todas las dependencias
pip install -r requirements.txt
```

**Dependencias principales instaladas:**
- Django 5.2.7
- TensorFlow 2.20.0
- Keras 3.12.0
- Pillow (procesamiento de imágenes)
- python-dotenv (variables de entorno)
- psycopg2-binary (PostgreSQL)
- django-cors-headers (API CORS)
- supabase (cliente de Supabase)

#### 4️⃣ Configurar Variables de Entorno

El proyecto incluye un archivo `.env.example` con todas las variables necesarias. Este archivo sirve como plantilla y **nunca debe contener valores reales** (ya está en `.gitignore`).

**Windows (PowerShell/CMD):**
```powershell
# Copiar archivo de ejemplo
copy .env.example .env

# Editar con tu editor favorito
notepad .env
# o
code .env  # Si tienes VS Code
```

**Linux/Mac:**
```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar con tu editor favorito
nano .env
# o
vim .env
```

**📝 Contenido del archivo `.env` (ejemplo completo):**

```env
# ==========================================
# DJANGO CONFIGURATION
# ==========================================
SECRET_KEY='django-insecure-change-this-to-a-random-secret-key'
DEBUG=True  # Cambiar a False en producción
ALLOWED_HOSTS=localhost,127.0.0.1

# ==========================================
# SUPABASE CONFIGURATION (Opcional)
# Solo necesario si usas Supabase en producción
# ==========================================
SUPABASE_URL=https://tu-proyecto.supabase.co
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# ==========================================
# DATABASE CONFIGURATION
# Por defecto usa SQLite (no requiere configuración)
# Descomenta las siguientes líneas para usar PostgreSQL
# ==========================================
# DATABASE_URL=postgresql://postgres:password@host:port/database
# DB_NAME=postgres
# DB_USER=postgres.projectid  # Para Supabase pooler
# DB_PASSWORD=tu-password-seguro
# DB_HOST=aws-0-us-west-1.pooler.supabase.com
# DB_PORT=6543

# ==========================================
# MEDIA & STATIC FILES
# ==========================================
MEDIA_ROOT=media/
MEDIA_URL=/media/
STATIC_ROOT=staticfiles/
STATIC_URL=/static/

# ==========================================
# MODEL CONFIGURATION
# ==========================================
MODEL_PATH=models/improved_balanced_7class_model.h5
IMAGE_SIZE=224

# ==========================================
# APPLICATION SETTINGS
# ==========================================
MAX_UPLOAD_SIZE=10485760  # 10MB en bytes
ALLOWED_IMAGE_EXTENSIONS=jpg,jpeg,png
FILE_UPLOAD_MAX_MEMORY_SIZE=10485760
DATA_UPLOAD_MAX_MEMORY_SIZE=10485760

# ==========================================
# EMAIL CONFIGURATION (Opcional)
# Para envío de reportes y recuperación de contraseña
# ==========================================
# EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
# EMAIL_HOST=smtp.gmail.com
# EMAIL_PORT=587
# EMAIL_USE_TLS=True
# EMAIL_HOST_USER=tu-email@gmail.com
# EMAIL_HOST_PASSWORD=tu-app-password-de-gmail
# DEFAULT_FROM_EMAIL=SkinAI <tu-email@gmail.com>

# ==========================================
# GOOGLE OAUTH2 (Opcional)
# Necesario si quieres login con Google
# ==========================================
# GOOGLE_OAUTH2_CLIENT_ID=tu-client-id.apps.googleusercontent.com
# GOOGLE_OAUTH2_CLIENT_SECRET=tu-client-secret
```

**⚠️ IMPORTANTE - Seguridad del archivo `.env`:**

1. **NUNCA** subas el archivo `.env` a GitHub (ya está en `.gitignore`)
2. **Genera una nueva SECRET_KEY** única para tu proyecto (ver sección siguiente)
3. **Cambia DEBUG=False** en producción
4. **No compartas** tus credenciales de Supabase o email
5. Cada desarrollador debe tener su **propio archivo `.env`** con sus credenciales locales

#### 5️⃣ Configurar Base de Datos

```bash
# Crear migraciones
python manage.py makemigrations

# Aplicar migraciones
python manage.py migrate

# Crear superusuario (admin)
python manage.py createsuperuser
```

#### 6️⃣ Colectar Archivos Estáticos

```bash
python manage.py collectstatic --noinput
```

#### 7️⃣ Ejecutar Servidor de Desarrollo

```bash
python manage.py runserver

# El servidor estará disponible en:
# http://127.0.0.1:8000
```

### Variables de Entorno

#### 📝 Descripción de Variables

El archivo `.env` contiene todas las configuraciones sensibles del proyecto. A continuación, la descripción completa de cada variable:

##### 🔧 Django Core

| Variable | Descripción | Requerido | Ejemplo |
|----------|-------------|-----------|---------|
| `SECRET_KEY` | Clave secreta de Django para seguridad criptográfica. **Debe ser única y secreta** | ✅ Sí | `django-insecure-xyz123...` |
| `DEBUG` | Modo debug (True para desarrollo, False para producción) | ✅ Sí | `True` |
| `ALLOWED_HOSTS` | Hosts permitidos separados por coma | ✅ Sí | `localhost,127.0.0.1` |

##### ☁️ Supabase (Opcional - Solo Producción)

| Variable | Descripción | Requerido | Ejemplo |
|----------|-------------|-----------|---------|
| `SUPABASE_URL` | URL de tu proyecto Supabase | ❌ No | `https://xxx.supabase.co` |
| `SUPABASE_ANON_KEY` | Anon key pública de Supabase | ❌ No | `eyJhbGci...` |
| `SUPABASE_SERVICE_KEY` | Service key privada de Supabase (solo backend) | ❌ No | `eyJhbGci...` |

##### 🗄️ Base de Datos

| Variable | Descripción | Requerido | Ejemplo |
|----------|-------------|-----------|---------|
| `DATABASE_URL` | URL completa de conexión PostgreSQL (formato URI) | ❌ No | `postgresql://user:pass@host:port/db` |
| `DB_NAME` | Nombre de la base de datos | ❌ No | `postgres` |
| `DB_USER` | Usuario de la base de datos | ❌ No | `postgres.projectid` |
| `DB_PASSWORD` | Contraseña de la base de datos | ❌ No | `tu_password` |
| `DB_HOST` | Host de la base de datos | ❌ No | `aws-0-us-west-1.pooler.supabase.com` |
| `DB_PORT` | Puerto de la base de datos | ❌ No | `6543` o `5432` |

> **💡 Nota:** Si no configuras estas variables, el proyecto usará **SQLite** automáticamente (perfecto para desarrollo local).

##### 📁 Archivos y Media

| Variable | Descripción | Requerido | Ejemplo |
|----------|-------------|-----------|---------|
| `MEDIA_ROOT` | Directorio para archivos subidos por usuarios | ✅ Sí | `media/` |
| `MEDIA_URL` | URL base para servir archivos de media | ✅ Sí | `/media/` |
| `STATIC_ROOT` | Directorio para archivos estáticos en producción | ❌ No | `staticfiles/` |
| `STATIC_URL` | URL base para archivos estáticos | ✅ Sí | `/static/` |

##### 🧠 Modelo de IA

| Variable | Descripción | Requerido | Ejemplo |
|----------|-------------|-----------|---------|
| `MODEL_PATH` | Ruta del archivo del modelo TensorFlow/Keras | ✅ Sí | `models/improved_balanced_7class_model.h5` |
| `IMAGE_SIZE` | Tamaño de entrada del modelo (píxeles) | ✅ Sí | `224` |

##### � Límites de Subida

| Variable | Descripción | Requerido | Ejemplo |
|----------|-------------|-----------|---------|
| `MAX_UPLOAD_SIZE` | Tamaño máximo de archivo en bytes | ❌ No | `10485760` (10 MB) |
| `ALLOWED_IMAGE_EXTENSIONS` | Extensiones permitidas separadas por coma | ❌ No | `jpg,jpeg,png` |
| `FILE_UPLOAD_MAX_MEMORY_SIZE` | Límite de memoria para uploads | ❌ No | `10485760` |
| `DATA_UPLOAD_MAX_MEMORY_SIZE` | Límite de memoria para datos POST | ❌ No | `10485760` |

##### 📧 Email (Opcional)

| Variable | Descripción | Requerido | Ejemplo |
|----------|-------------|-----------|---------|
| `EMAIL_BACKEND` | Backend de email de Django | ❌ No | `django.core.mail.backends.smtp.EmailBackend` |
| `EMAIL_HOST` | Servidor SMTP | ❌ No | `smtp.gmail.com` |
| `EMAIL_PORT` | Puerto SMTP | ❌ No | `587` |
| `EMAIL_USE_TLS` | Usar TLS para conexión segura | ❌ No | `True` |
| `EMAIL_HOST_USER` | Email del remitente | ❌ No | `tu-email@gmail.com` |
| `EMAIL_HOST_PASSWORD` | Contraseña o App Password | ❌ No | `tu_app_password` |
| `DEFAULT_FROM_EMAIL` | Remitente por defecto | ❌ No | `SkinAI <email@gmail.com>` |

##### 🔐 OAuth2 (Opcional)

| Variable | Descripción | Requerido | Ejemplo |
|----------|-------------|-----------|---------|
| `GOOGLE_OAUTH2_CLIENT_ID` | Client ID de Google Cloud Console | ❌ No | `xxx.apps.googleusercontent.com` |
| `GOOGLE_OAUTH2_CLIENT_SECRET` | Client Secret de Google | ❌ No | `GOCSPX-xxx` |

#### 🔐 Generar SECRET_KEY

Django requiere una clave secreta única. **Nunca uses la del ejemplo en producción**:

**Método 1: Comando Python**
```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

**Método 2: Online (Desarrollo)** 
- Visita: https://djecrety.ir/
- Copia la clave generada

**Método 3: Python Shell**
```bash
python manage.py shell
```
```python
from django.core.management.utils import get_random_secret_key
print(get_random_secret_key())
exit()
```

**Reemplaza en `.env`:**
```env
SECRET_KEY='tu-nueva-secret-key-generada-aqui'
```

#### 📋 Archivo `.env.example`

El proyecto incluye `.env.example` como plantilla. Este archivo:
- ✅ **Está en el repositorio** (seguro para compartir)
- ✅ **No contiene credenciales reales**
- ✅ **Documenta todas las variables disponibles**
- ✅ **Usa valores de ejemplo/placeholder**

Para usarlo:
```bash
# Copia el ejemplo
cp .env.example .env

# Edita con tus valores reales
nano .env

# El archivo .env está en .gitignore (no se subirá a GitHub)
```

#### ⚠️ Checklist de Seguridad

Antes de desplegar a producción, verifica:

- [ ] `SECRET_KEY` es única y diferente al ejemplo
- [ ] `DEBUG=False` en producción
- [ ] `ALLOWED_HOSTS` incluye tu dominio de producción
- [ ] Credenciales de base de datos son seguras
- [ ] Archivo `.env` **NO** está en el repositorio
- [ ] `.env` está listado en `.gitignore`
- [ ] Variables sensibles no están en código fuente
- [ ] `EMAIL_HOST_PASSWORD` es un App Password (no la contraseña real)
- [ ] OAuth2 secrets están protegidos

---

## 📝 Guía de Uso

### Para Usuarios

#### 1. Registrarse en el Sistema

1. Accede a http://127.0.0.1:8000
2. Haz clic en **"Registrarse"** en la barra de navegación
3. Completa el formulario:
   - Nombre de usuario
   - Email
   - Contraseña (mínimo 8 caracteres)
   - Confirmar contraseña
4. Haz clic en **"Crear Cuenta"**

**O usa Google Sign-In:**
- Haz clic en "Iniciar sesión con Google"
- Autoriza la aplicación con tu cuenta de Google

#### 2. Realizar un Diagnóstico

1. **Iniciar Sesión**
   - Ve a http://127.0.0.1:8000/login/
   - Ingresa tus credenciales

2. **Subir Imagen**
   - Desde el dashboard, haz clic en **"Nuevo Diagnóstico"**
   - Selecciona una imagen de tu dispositivo:
     - Formatos aceptados: JPG, JPEG, PNG
     - Tamaño máximo: 10 MB
     - Resolución recomendada: >200x200 píxeles
   - O toma una foto con la cámara (en dispositivos móviles)

3. **Ver Resultados**
   - El sistema procesará la imagen (2-5 segundos)
   - Verás:
     - ✅ **Diagnóstico principal** con porcentaje de confianza
     - 📊 **Gráfico de probabilidades** de todas las clases
     - 📋 **Información detallada** de la enfermedad detectada
     - ⚠️ **Recomendaciones médicas**
     - 📝 **Nivel de riesgo**

4. **Descargar/Compartir Reporte**
   - Haz clic en **"Descargar PDF"** para obtener un reporte completo
   - Usa **"Enviar por Email"** para recibir el reporte en tu correo

#### 3. Consultar Historial

1. Ve a **"Mi Historial"** en el menú
2. Verás todos tus diagnósticos anteriores:
   - Fecha y hora de análisis
   - Imagen analizada (miniatura)
   - Resultado del diagnóstico
   - Nivel de confianza
3. Haz clic en cualquier diagnóstico para ver detalles completos

#### 4. Explorar Información de Enfermedades

1. Ve a **"Enfermedades"** en el menú
2. Selecciona cualquier tipo de lesión
3. Aprende sobre:
   - Descripción médica
   - Síntomas característicos
   - Causas comunes
   - Factores de riesgo
   - Tratamientos disponibles
   - Pronóstico

### Para Desarrolladores

#### Ejecutar en Modo Desarrollo

```bash
# Activar entorno virtual
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Ejecutar servidor con debug
python manage.py runserver 0.0.0.0:8000

# O con auto-reload para desarrollo
python manage.py runserver --noreload
```

#### Acceder al Panel de Administración

1. Crea un superusuario (si no lo has hecho):
```bash
python manage.py createsuperuser
```

2. Accede a http://127.0.0.1:8000/admin/
3. Inicia sesión con tus credenciales de superusuario
4. Podrás administrar:
   - Usuarios y permisos
   - Predicciones almacenadas
   - Cuentas sociales (OAuth)
   - Configuración del sistema

#### Ejecutar Tests

```bash
# Ejecutar todos los tests
python manage.py test

# Ejecutar tests de una app específica
python manage.py test skin_detector

# Con cobertura
pip install coverage
coverage run --source='.' manage.py test
coverage report
```

#### Verificar Modelo de IA

```bash
# Probar carga del modelo
python test_model_loading.py

# Verificar compatibilidad de NumPy
python test_numpy_compatibility.py

# Probar conexión a Supabase (opcional)
python test_supabase_connection.py
```

---

## 📚 Guía Técnica

### Arquitectura

#### Patrón MTV (Model-Template-View)

```
┌─────────────────────────────────────────────────────────────┐
│                    FLUJO DE LA APLICACIÓN                    │
└─────────────────────────────────────────────────────────────┘

Usuario (Navegador/App Móvil)
    │
    │ HTTP Request (GET/POST)
    ↓
┌─────────────────────┐
│  URLS.PY           │ → Enrutador de URLs
│  (URLconf)         │   Mapea URLs a vistas
└─────────────────────┘
    │
    ↓
┌─────────────────────┐
│  VIEWS.PY          │ → Lógica de Negocio
│  (Controllers)     │   Procesa requests
└─────────────────────┘   Ejecuta predicciones
    │          │          Prepara contexto
    │          │
    ↓          ↓
┌──────────┐  ┌────────────┐
│ MODELS   │  │ TEMPLATES  │
│ (DB ORM) │  │ (HTML+DTL) │
└──────────┘  └────────────┘
    │              │
    ↓              ↓
DATABASE      HTML Response
```

#### Componentes Principales

**1. Django Backend:**
- **Views**: Controladores que procesan requests
- **Models**: ORM para interactuar con la base de datos
- **Forms**: Validación de formularios
- **URLs**: Sistema de enrutamiento

**2. Modelo de IA:**
- **TensorFlow/Keras**: Motor de inferencia
- **Predictor**: Clase wrapper para el modelo
- **Preprocessing**: Normalización y redimensionamiento de imágenes

**3. Base de Datos:**
- **SQLite**: Para desarrollo local
- **PostgreSQL (Supabase)**: Para producción
- **Migraciones**: Control de versiones del schema

**4. Frontend:**
- **Django Templates**: HTML dinámico
- **Bootstrap 5**: Framework CSS
- **JavaScript**: Interactividad y validación

### Estructura del Proyecto

```
django_skin_disease_detector/
│
├── 📂 skin_disease_project/       # Configuración del proyecto
│   ├── settings.py                # Configuración principal
│   ├── urls.py                    # URLs raíz
│   └── wsgi.py                    # WSGI para producción
│
├── 📂 skin_detector/              # App principal
│   ├── 📄 models.py               # Modelos de datos
│   │   ├── SkinImagePrediction    # Predicciones
│   │   └── SocialAccount          # OAuth2
│   │
│   ├── 📄 views.py                # Vistas/Controladores
│   │   ├── landing_view()         # Página de inicio
│   │   ├── home_view()            # Dashboard
│   │   ├── predict_view()         # Subida y predicción
│   │   ├── prediction_detail()    # Resultados
│   │   ├── history_view()         # Historial
│   │   └── [Auth views...]        # Registro/Login
│   │
│   ├── 📄 predictor.py            # Motor de IA
│   │   └── SkinDiseasePredictor   # Clase principal
│   │       ├── load_model()
│   │       ├── preprocess_image()
│   │       └── predict()
│   │
│   ├── 📄 forms.py                # Formularios Django
│   │   ├── ImageUploadForm
│   │   ├── UserRegistrationForm
│   │   └── UserLoginForm
│   │
│   ├── 📄 urls.py                 # URLs de la app
│   ├── 📄 constants.py            # Info de enfermedades
│   ├── 📄 utils.py                # Funciones auxiliares
│   └── 📂 migrations/             # Migraciones DB
│
├── 📂 templates/                  # Plantillas HTML
│   └── skin_detector/
│       ├── base.html              # Plantilla base
│       ├── landing.html           # Landing page
│       ├── home.html              # Dashboard
│       ├── prediction_detail.html # Resultados
│       ├── history.html           # Historial
│       ├── login.html             # Login
│       └── register.html          # Registro
│
├── 📂 static/                     # Archivos estáticos
│   ├── css/style.css              # Estilos
│   ├── js/main.js                 # JavaScript
│   └── images/                    # Imágenes
│
├── 📂 media/                      # Uploads de usuarios
│   └── skin_images/               # Imágenes de lesiones
│
├── 📂 models/                     # Modelos de IA
│   └── improved_balanced_7class_model.h5
│
├── 📄 manage.py                   # CLI de Django
├── 📄 requirements.txt            # Dependencias
├── 📄 .env                        # Variables de entorno
└── 📄 README.md                   # Este archivo
```

### API REST

#### Endpoints Disponibles

**Base URL:** `http://127.0.0.1:8000/api/`

#### 1. Predicción de Imágenes

```http
POST /api/predict/
Content-Type: multipart/form-data

# Body (form-data)
image: [archivo de imagen]
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "id": 123,
    "disease": "MEL",
    "disease_name": "Melanoma",
    "confidence": 0.87,
    "probabilities": {
      "MEL": 0.87,
      "NV": 0.05,
      "BCC": 0.03,
      "AKIEC": 0.02,
      "BKL": 0.01,
      "DF": 0.01,
      "VASC": 0.01
    },
    "risk_level": "high",
    "created_at": "2025-11-06T10:30:00Z"
  }
}
```

#### 2. Obtener Historial

```http
GET /api/history/
Authorization: Token <your-token-here>
```

**Response:**
```json
{
  "success": true,
  "count": 15,
  "results": [
    {
      "id": 123,
      "disease": "MEL",
      "confidence": 0.87,
      "created_at": "2025-11-06T10:30:00Z",
      "image_url": "/media/skin_images/image_123.jpg"
    },
    ...
  ]
}
```

#### 3. Obtener Detalle de Predicción

```http
GET /api/predictions/<id>/
Authorization: Token <your-token-here>
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "id": 123,
    "disease": "MEL",
    "disease_name": "Melanoma",
    "confidence": 0.87,
    "description": "El melanoma es...",
    "symptoms": ["Asimetría", "Bordes irregulares", ...],
    "treatment": "Extirpación quirúrgica...",
    "created_at": "2025-11-06T10:30:00Z"
  }
}
```

#### Autenticación

La API usa **Token Authentication** de Django REST Framework:

```bash
# Obtener token
POST /api/auth/login/
{
  "username": "usuario",
  "password": "contraseña"
}

# Response
{
  "token": "9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b"
}

# Usar token en requests
curl -H "Authorization: Token 9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b" \
     http://127.0.0.1:8000/api/history/
```

### Modelos de Datos

#### SkinImagePrediction

```python
class SkinImagePrediction(models.Model):
    user = ForeignKey(User)              # Usuario que subió la imagen
    image = ImageField()                 # Imagen de la lesión
    predicted_class = CharField()        # Clase predicha (MEL, NV, etc.)
    confidence = FloatField()            # Confianza (0.0-1.0)
    probabilities = JSONField()          # Probabilidades de todas las clases
    created_at = DateTimeField()         # Fecha de creación
    updated_at = DateTimeField()         # Última actualización
```

#### SocialAccount (OAuth2)

```python
class SocialAccount(models.Model):
    user = ForeignKey(User)              # Usuario asociado
    provider = CharField()               # Proveedor (google, facebook)
    provider_user_id = CharField()       # ID del usuario en el proveedor
    access_token = TextField()           # Token de acceso
    refresh_token = TextField()          # Token de refresco
    token_expires_at = DateTimeField()   # Expiración del token
```

---

## 🔧 Configuración Avanzada

### Usar PostgreSQL en Local

1. Instalar PostgreSQL:
```bash
# Windows: Descarga desde https://www.postgresql.org/download/
# Linux (Ubuntu/Debian):
sudo apt update
sudo apt install postgresql postgresql-contrib
# Mac:
brew install postgresql
```

2. Crear base de datos:
```sql
sudo -u postgres psql
CREATE DATABASE skinai_db;
CREATE USER skinai_user WITH PASSWORD 'tu_password';
GRANT ALL PRIVILEGES ON DATABASE skinai_db TO skinai_user;
\q
```

3. Actualizar `settings.py`:
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'skinai_db',
        'USER': 'skinai_user',
        'PASSWORD': 'tu_password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
```

4. Migrar:
```bash
python manage.py migrate
```

### Configurar Email (SMTP)

Para envío de reportes y recuperación de contraseña:

```python
# En settings.py
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = 'tu_email@gmail.com'
EMAIL_HOST_PASSWORD = 'tu_app_password'  # No uses la contraseña real
DEFAULT_FROM_EMAIL = 'SkinAI <tu_email@gmail.com>'
```

### Configurar Google OAuth2

1. Ve a [Google Cloud Console](https://console.cloud.google.com/)
2. Crea un nuevo proyecto
3. Habilita "Google+ API"
4. Crea credenciales OAuth 2.0:
   - Application type: Web application
   - Authorized redirect URIs: `http://localhost:8000/auth/google/callback/`
5. Copia Client ID y Client Secret
6. Actualiza `settings.py`:
```python
GOOGLE_OAUTH_CLIENT_ID = 'tu-client-id.apps.googleusercontent.com'
GOOGLE_OAUTH_CLIENT_SECRET = 'tu-client-secret'
```

### Optimizar Rendimiento

#### 1. Cache con Redis

```bash
pip install redis django-redis
```

```python
# settings.py
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}
```

#### 2. Celery para Tareas Asíncronas

```bash
pip install celery redis
```

```python
# celery.py
from celery import Celery

app = Celery('skinai')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

@app.task
def process_prediction_async(image_path, user_id):
    # Procesar predicción en background
    pass
```

---

## 🧪 Testing

### Ejecutar Tests Unitarios

```bash
# Todos los tests
python manage.py test

# Tests específicos
python manage.py test skin_detector.tests.TestPredictor
python manage.py test skin_detector.tests.TestViews
python manage.py test skin_detector.tests.TestModels

# Con verbosidad
python manage.py test --verbosity=2

# Mantener base de datos de test
python manage.py test --keepdb
```

### Tests de Integración

```bash
# Test del modelo de IA
python test_model_loading.py

# Test de compatibilidad NumPy/TensorFlow
python test_numpy_compatibility.py

# Test de conexión a Supabase
python test_supabase_connection.py

# Test de conexión a base de datos
python test_db_connection.py
```

### Coverage (Cobertura de Código)

```bash
# Instalar coverage
pip install coverage

# Ejecutar tests con coverage
coverage run --source='skin_detector' manage.py test

# Ver reporte en terminal
coverage report

# Generar reporte HTML
coverage html
# Abre htmlcov/index.html en el navegador
```

### Tests Manuales

**Checklist de pruebas:**

- [ ] Registro de usuario funciona
- [ ] Login con credenciales funciona
- [ ] Login con Google funciona
- [ ] Subida de imagen funciona
- [ ] Predicción retorna resultados correctos
- [ ] Gráficos de probabilidades se muestran
- [ ] Historial muestra predicciones anteriores
- [ ] Información de enfermedades es correcta
- [ ] Descarga de PDF funciona
- [ ] Envío de email funciona
- [ ] Logout funciona
- [ ] Recuperación de contraseña funciona

---

## 🚀 Despliegue

### Despliegue en Heroku

#### 1. Preparar para Producción

```python
# settings.py
DEBUG = False
ALLOWED_HOSTS = ['tu-app.herokuapp.com']
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
```

#### 2. Crear archivos necesarios

**Procfile:**
```
web: gunicorn skin_disease_project.wsgi --log-file -
```

**runtime.txt:**
```
python-3.11.5
```

**requirements.txt:**
```bash
pip freeze > requirements.txt
```

#### 3. Instalar Heroku CLI y Deploy

```bash
# Instalar Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Login
heroku login

# Crear app
heroku create tu-app-name

# Agregar PostgreSQL
heroku addons:create heroku-postgresql:hobby-dev

# Configurar variables de entorno
heroku config:set SECRET_KEY='tu-secret-key'
heroku config:set DEBUG=False
heroku config:set MODEL_PATH='models/improved_balanced_7class_model.h5'

# Deploy
git push heroku main

# Migrar base de datos
heroku run python manage.py migrate

# Crear superusuario
heroku run python manage.py createsuperuser

# Abrir app
heroku open
```

### Despliegue en Railway

#### 1. Crear cuenta en [Railway.app](https://railway.app/)

#### 2. Crear nuevo proyecto desde GitHub

#### 3. Configurar variables de entorno en Railway

```
SECRET_KEY=tu-secret-key
DEBUG=False
ALLOWED_HOSTS=tu-app.railway.app
DATABASE_URL=postgresql://...
```

#### 4. Railway detectará automáticamente Django y lo desplegará

### Despliegue en DigitalOcean

#### 1. Crear Droplet (Ubuntu 22.04)

#### 2. Configurar servidor

```bash
# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar dependencias
sudo apt install python3-pip python3-venv nginx postgresql postgresql-contrib -y

# Crear usuario
sudo adduser skinai
sudo usermod -aG sudo skinai
su - skinai

# Clonar repositorio
git clone https://github.com/tu-usuario/Skin-lesion-analyzer.git
cd Skin-lesion-analyzer/django_skin_disease_detector

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
pip install gunicorn

# Configurar PostgreSQL
sudo -u postgres psql
CREATE DATABASE skinai_db;
CREATE USER skinai_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE skinai_db TO skinai_user;
\q

# Configurar .env
nano .env
# Agregar variables de entorno

# Migrar
python manage.py migrate
python manage.py collectstatic

# Crear superusuario
python manage.py createsuperuser
```

#### 3. Configurar Gunicorn

```bash
# Crear archivo de servicio
sudo nano /etc/systemd/system/gunicorn.service
```

```ini
[Unit]
Description=gunicorn daemon for SkinAI
After=network.target

[Service]
User=skinai
Group=www-data
WorkingDirectory=/home/skinai/Skin-lesion-analyzer/django_skin_disease_detector
ExecStart=/home/skinai/Skin-lesion-analyzer/django_skin_disease_detector/venv/bin/gunicorn \
          --workers 3 \
          --bind unix:/home/skinai/Skin-lesion-analyzer/django_skin_disease_detector/skinai.sock \
          skin_disease_project.wsgi:application

[Install]
WantedBy=multi-user.target
```

```bash
# Iniciar servicio
sudo systemctl start gunicorn
sudo systemctl enable gunicorn
```

#### 4. Configurar Nginx

```bash
sudo nano /etc/nginx/sites-available/skinai
```

```nginx
server {
    listen 80;
    server_name tu-dominio.com;

    location /static/ {
        alias /home/skinai/Skin-lesion-analyzer/django_skin_disease_detector/staticfiles/;
    }

    location /media/ {
        alias /home/skinai/Skin-lesion-analyzer/django_skin_disease_detector/media/;
    }

    location / {
        include proxy_params;
        proxy_pass http://unix:/home/skinai/Skin-lesion-analyzer/django_skin_disease_detector/skinai.sock;
    }
}
```

```bash
# Activar sitio
sudo ln -s /etc/nginx/sites-available/skinai /etc/nginx/sites-enabled
sudo nginx -t
sudo systemctl restart nginx
```

#### 5. Configurar SSL con Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d tu-dominio.com
```

---

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Aquí está cómo puedes ayudar:

### Reportar Bugs

1. Ve a [Issues](https://github.com/ecx567/Skin-lesion-analyzer/issues)
2. Crea un nuevo issue
3. Describe el bug detalladamente:
   - Pasos para reproducir
   - Comportamiento esperado
   - Comportamiento actual
   - Screenshots (si aplica)
   - Información del sistema (OS, Python version, etc.)

### Sugerir Mejoras

1. Abre un issue con la etiqueta "enhancement"
2. Describe la mejora que propones
3. Explica por qué sería útil
4. Proporciona ejemplos si es posible

### Contribuir Código

1. **Fork** el repositorio
2. **Crea una rama** para tu feature:
```bash
git checkout -b feature/mi-nueva-funcionalidad
```

3. **Haz commit** de tus cambios:
```bash
git commit -m "feat: Agregar nueva funcionalidad X"
```

4. **Push** a tu rama:
```bash
git push origin feature/mi-nueva-funcionalidad
```

5. **Abre un Pull Request** en GitHub

### Convenciones de Código

- Sigue [PEP 8](https://pep8.org/) para Python
- Usa nombres descriptivos para variables y funciones
- Documenta funciones complejas con docstrings
- Escribe tests para nuevas funcionalidades
- Actualiza el README si es necesario

### Commits Semánticos

Usamos [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: Nueva característica
fix: Corrección de bug
docs: Cambios en documentación
style: Cambios de formato (no afectan funcionalidad)
refactor: Refactorización de código
test: Agregar o modificar tests
chore: Tareas de mantenimiento
```

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

```
MIT License

Copyright (c) 2025 SkinAI Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 👥 Equipo

- **Desarrolladores**: [Contribuidores](https://github.com/ecx567/Skin-lesion-analyzer/graphs/contributors)
- **Mantenedor**: [@ecx567](https://github.com/ecx567)

---

## 📞 Contacto y Soporte

- **GitHub Issues**: [Reportar problema](https://github.com/ecx567/Skin-lesion-analyzer/issues)
- **Email**: support@skinai.example.com
- **Documentación**: [Wiki del proyecto](https://github.com/ecx567/Skin-lesion-analyzer/wiki)

---

## 🙏 Agradecimientos

- **HAM10000 Dataset** - Por proporcionar el dataset de entrenamiento
- **TensorFlow Team** - Por el framework de Deep Learning
- **Django Community** - Por el excelente framework web
- **Supabase** - Por la infraestructura de base de datos
- **Todos los contribuidores** - Por hacer este proyecto posible

---

## 📚 Referencias

1. Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset. *Nature Scientific Data*, 5, 180161.
2. Esteva, A., et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks. *Nature*, 542, 115-118.
3. Codella, N., et al. (2018). Skin Lesion Analysis Toward Melanoma Detection. *ISIC Challenge*.

---

<div align="center">

**⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub ⭐**

[⬆ Volver arriba](#-skinai---sistema-de-detección-de-enfermedades-cutáneas)

</div>
