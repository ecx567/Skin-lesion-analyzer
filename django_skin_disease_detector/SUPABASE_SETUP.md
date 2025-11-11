# Configuración de Supabase - Estado Actual

## 🎯 Resumen
El sistema de detección de enfermedades cutáneas ahora está configurado con:
- ✅ **Django con SQLite** (Base de datos local)
- ✅ **Supabase configurado** (Acceso via MCP tools y REST API)
- ✅ **Tablas creadas en Supabase** (6 tablas con esquema completo)
- ⚠️ **Sin conexión directa PostgreSQL** (requiere plan pago de Supabase)

---

## 📋 Tablas Creadas en Supabase

### 1. `auth_user` (11 columnas, 0 rows)
Tabla de usuarios de Django:
```sql
- id (INTEGER PRIMARY KEY)
- username (VARCHAR(150) UNIQUE)
- email (VARCHAR(254))
- password (VARCHAR(128))
- first_name, last_name
- is_staff, is_active, is_superuser
- date_joined, last_login
```

### 2. `skin_detector_skinimageprediction` (10 columnas, 0 rows)
Tabla principal de predicciones:
```sql
- id (SERIAL PRIMARY KEY)
- image (VARCHAR(100))
- user_id (INTEGER FOREIGN KEY → auth_user)
- predicted_class (VARCHAR(50) CHECK: akiec|bcc|bkl|df|mel|nv|vasc)
- confidence_score (NUMERIC CHECK: 0-1)
- probabilities (JSONB)
- uploaded_at (TIMESTAMP DEFAULT NOW())
- notes (TEXT NULLABLE)
```

**Índices:**
- `idx_prediction_uploaded_at` (uploaded_at DESC)
- `idx_prediction_class` (predicted_class)
- `idx_prediction_user` (user_id)

**Constraints:**
- CHECK: predicted_class IN (7 valores válidos)
- CHECK: confidence_score BETWEEN 0 AND 1
- FOREIGN KEY: user_id → auth_user(id) ON DELETE CASCADE

### 3. `skin_detector_socialaccount` (6 columnas, 0 rows)
Autenticación social (Google, Facebook, etc.):
```sql
- id (SERIAL PRIMARY KEY)
- user_id (INTEGER FOREIGN KEY → auth_user)
- provider (VARCHAR(50))
- provider_user_id (VARCHAR(255))
- email (VARCHAR(254))
- extra_data (JSONB)
```

**Constraints:**
- UNIQUE (provider, provider_user_id)

### 4. `django_session` (3 columnas, 0 rows)
Sesiones de Django:
```sql
- session_key (VARCHAR(40) PRIMARY KEY)
- session_data (TEXT)
- expire_date (TIMESTAMP)
```

**Índices:**
- `idx_session_expire_date` (expire_date)

### 5. `django_content_type` (3 columnas, 0 rows)
Content Types de Django:
```sql
- id (SERIAL PRIMARY KEY)
- app_label (VARCHAR(100))
- model (VARCHAR(100))
```

**Constraints:**
- UNIQUE (app_label, model)

### 6. `django_migrations` (3 columnas, 0 rows)
Historial de migraciones:
```sql
- id (SERIAL PRIMARY KEY)
- app (VARCHAR(255))
- name (VARCHAR(255))
- applied (TIMESTAMP DEFAULT NOW())
```

---

## 🔧 Funciones SQL Helper

### `get_user_prediction_stats(user_id)`
Retorna estadísticas de predicciones por usuario:
```sql
SELECT 
    p_user_id,
    COUNT(*) as total_predictions,
    COUNT(DISTINCT predicted_class) as unique_classes,
    AVG(confidence_score) as avg_confidence,
    MAX(uploaded_at) as last_prediction_date
FROM skin_detector_skinimageprediction
WHERE user_id = p_user_id
GROUP BY p_user_id;
```

### `get_recent_predictions(limit)`
Retorna predicciones recientes con información de usuario:
```sql
SELECT 
    p.id,
    p.predicted_class,
    p.confidence_score,
    p.uploaded_at,
    u.username,
    u.email
FROM skin_detector_skinimageprediction p
INNER JOIN auth_user u ON p.user_id = u.id
ORDER BY p.uploaded_at DESC
LIMIT p_limit;
```

---

## 🚀 Cómo Usar Supabase con Django

### Opción 1: MCP Tools (Actual)
Usa las herramientas MCP para interactuar con Supabase:

```python
# En lugar de Django ORM, usar MCP tools:
from supabase import create_client

supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_ANON_KEY')
)

# Crear predicción
result = supabase.table('skin_detector_skinimageprediction').insert({
    'user_id': user.id,
    'predicted_class': 'mel',
    'confidence_score': 0.85,
    'probabilities': {'mel': 0.85, 'nv': 0.10, ...}
}).execute()

# Obtener predicciones
predictions = supabase.table('skin_detector_skinimageprediction')\
    .select('*')\
    .eq('user_id', user.id)\
    .order('uploaded_at', desc=True)\
    .execute()
```

### Opción 2: REST API
```python
import requests

headers = {
    'apikey': os.getenv('SUPABASE_ANON_KEY'),
    'Content-Type': 'application/json'
}

# POST para crear
response = requests.post(
    f"{os.getenv('SUPABASE_URL')}/rest/v1/skin_detector_skinimageprediction",
    headers=headers,
    json={
        'user_id': user.id,
        'predicted_class': 'mel',
        'confidence_score': 0.85,
        'probabilities': {'mel': 0.85, 'nv': 0.10}
    }
)

# GET para consultar
response = requests.get(
    f"{os.getenv('SUPABASE_URL')}/rest/v1/skin_detector_skinimageprediction",
    headers=headers,
    params={'user_id': f'eq.{user.id}', 'order': 'uploaded_at.desc'}
)
```

### Opción 3: PostgreSQL Directo (Requiere Plan Pago)
Para conexión directa desde Django, necesitas:
1. Plan pago de Supabase con **Direct Connection**
2. IP whitelisting
3. Certificados SSL

---

## ⚙️ Configuración Actual

### `.env`
```env
# Supabase REST API (funciona)
SUPABASE_URL=https://cpjmodytpeuybpcayzwk.supabase.co
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# PostgreSQL directo (no accesible sin plan pago)
DB_HOST=db.cpjmodytpeuybpcayzwk.supabase.co
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres.cpjmodytpeuybpcayzwk
DB_PASSWORD=database/skin|12345
```

### `settings.py`
```python
# SQLite para Django ORM (desarrollo local)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Supabase para API REST (producción)
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY')
```

---

## 📊 Estado de Migraciones

### Migraciones Aplicadas en Supabase
```
ID | App           | Migration Name                           | Applied
---+---------------+-----------------------------------------+---------
1  | skin_detector | create_auth_user_table                  | ✅
2  | skin_detector | create_skin_image_prediction_table      | ✅
3  | skin_detector | create_social_account_table             | ✅
4  | django        | create_django_session_table             | ✅
5  | django        | create_django_content_type_table        | ✅
6  | django        | create_django_migrations_table          | ✅
7  | functions     | create_helper_functions                 | ✅
```

Total: **11 migraciones** (5 antiguas + 6 nuevas)

---

## 🔍 Verificación

### Listar Tablas
```python
# Via MCP tool
mcp_supabase_list_tables(schemas=["public"])
# Output: 6 tablas con esquema completo
```

### Ver Datos
```python
# Via MCP tool
mcp_supabase_execute_sql(
    sql="SELECT * FROM auth_user LIMIT 10"
)
```

### Verificar Funciones
```sql
-- En Supabase SQL Editor
SELECT * FROM get_user_prediction_stats(1);
SELECT * FROM get_recent_predictions(10);
```

---

## 🎯 Próximos Pasos

### Implementación de Sincronización
Para sincronizar datos entre SQLite (Django) y Supabase:

1. **Crear middleware** para sincronizar automáticamente
2. **Usar signals** de Django para enviar a Supabase
3. **Background tasks** con Celery/Redis

Ejemplo con Django Signals:
```python
# skin_detector/signals.py
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import SkinImagePrediction
import requests
import os

@receiver(post_save, sender=SkinImagePrediction)
def sync_to_supabase(sender, instance, created, **kwargs):
    if created:
        # Enviar a Supabase
        headers = {
            'apikey': os.getenv('SUPABASE_ANON_KEY'),
            'Content-Type': 'application/json'
        }
        
        data = {
            'id': instance.id,
            'user_id': instance.user_id,
            'predicted_class': instance.predicted_class,
            'confidence_score': float(instance.confidence_score),
            'probabilities': instance.probabilities,
            'uploaded_at': instance.uploaded_at.isoformat()
        }
        
        requests.post(
            f"{os.getenv('SUPABASE_URL')}/rest/v1/skin_detector_skinimageprediction",
            headers=headers,
            json=data
        )
```

### Testing
```bash
# Verificar Django
python manage.py check

# Probar predicción
python manage.py runserver
# Upload image at http://localhost:8000

# Verificar en Supabase Dashboard
# https://cpjmodytpeuybpcayzwk.supabase.co
```

---

## 📚 Referencias

- [Supabase Documentation](https://supabase.com/docs)
- [Supabase Python Client](https://github.com/supabase-community/supabase-py)
- [Django Signals](https://docs.djangoproject.com/en/stable/topics/signals/)
- [PostgreSQL en Supabase](https://supabase.com/docs/guides/database)

---

## ⚠️ Notas Importantes

1. **Las tablas en Supabase están vacías** (0 rows) - necesitas sincronizar datos
2. **SQLite es para desarrollo** - Supabase es para producción via API REST
3. **Los índices y constraints están activos** - optimización de consultas
4. **Las funciones helper están disponibles** - estadísticas y queries
5. **No hay conexión directa PostgreSQL** - requiere plan pago de Supabase

---

**Fecha de creación:** 2025-11-09  
**Estado:** ✅ Configuración completa - Listo para desarrollo
