#!/usr/bin/env python3
"""
Script de verificación final de la integración
Verifica que todos los componentes están funcionando correctamente
"""
import os
import sys
import django

# Configurar Django
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'skin_disease_project.settings')
django.setup()

# Suprimir warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 80)
print("🔍 VERIFICACIÓN FINAL DE LA INTEGRACIÓN")
print("=" * 80)

# 1. Verificar versiones
print("\n1️⃣ Verificando versiones de dependencias...")
try:
    import tensorflow as tf
    import keras
    import numpy as np
    import django as dj
    
    print(f"   ✅ TensorFlow: {tf.__version__}")
    print(f"   ✅ Keras: {keras.__version__}")
    print(f"   ✅ NumPy: {np.__version__}")
    print(f"   ✅ Django: {dj.__version__}")
    
    # Verificar compatibilidad
    if keras.__version__.startswith('3.'):
        print(f"   ✅ Keras 3.x detectado - Compatible")
    else:
        print(f"   ⚠️  Keras {keras.__version__} - Se recomienda 3.x")
        
except ImportError as e:
    print(f"   ❌ Error importando dependencias: {e}")
    sys.exit(1)

# 2. Verificar archivo del modelo
print("\n2️⃣ Verificando archivo del modelo...")
from django.conf import settings

model_path = os.path.join(settings.BASE_DIR, 'models', 'improved_balanced_7class_model.h5')

if os.path.exists(model_path):
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   ✅ Modelo encontrado: {model_path}")
    print(f"   ✅ Tamaño: {file_size_mb:.2f} MB")
    
    if file_size_mb < 50:
        print(f"   ⚠️  Advertencia: El archivo parece pequeño para este modelo")
else:
    print(f"   ❌ Modelo NO encontrado: {model_path}")
    sys.exit(1)

# 3. Cargar predictor
print("\n3️⃣ Cargando predictor...")
try:
    from skin_detector.predictor import get_predictor
    
    predictor = get_predictor()
    print(f"   ✅ Predictor cargado exitosamente")
    
except Exception as e:
    print(f"   ❌ Error cargando predictor: {e}")
    import traceback
    print(traceback.format_exc())
    sys.exit(1)

# 4. Verificar resumen del modelo
print("\n4️⃣ Verificando resumen del modelo...")
try:
    summary = predictor.get_model_summary()
    
    if 'error' in summary:
        print(f"   ❌ Error: {summary['error']}")
        sys.exit(1)
    
    print(f"   ✅ Capas: {summary['total_layers']}")
    print(f"   ✅ Parámetros: {summary['total_params']:,}")
    print(f"   ✅ Input: {summary['input_shape']}")
    print(f"   ✅ Output: {summary['output_shape']}")
    print(f"   ✅ Clases: {len(summary['classes'])}")
    
    # Verificar que tiene 7 clases
    if len(summary['classes']) != 7:
        print(f"   ⚠️  Advertencia: Se esperaban 7 clases, se encontraron {len(summary['classes'])}")
    
except Exception as e:
    print(f"   ❌ Error obteniendo resumen: {e}")
    sys.exit(1)

# 5. Verificar información de clases
print("\n5️⃣ Verificando información de clases...")
expected_classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

print(f"   Clases del modelo:")
for idx, class_info in predictor.class_names.items():
    code = class_info['code']
    spanish = class_info['spanish']
    
    if code in expected_classes:
        print(f"   ✅ {idx}: {code} - {spanish}")
    else:
        print(f"   ⚠️  {idx}: {code} - NO ESPERADO")

# 6. Verificar información médica
print("\n6️⃣ Verificando información médica...")
for class_code in expected_classes:
    if class_code in predictor.disease_info:
        info = predictor.disease_info[class_code]
        print(f"   ✅ {class_code}: Severidad {info['severity']}, Riesgo {info['risk_level']}")
    else:
        print(f"   ❌ {class_code}: Información médica faltante")

# 7. Verificar archivos estáticos
print("\n7️⃣ Verificando archivos estáticos...")
static_files = [
    'static/css/style.css',
    'static/css/style_improved.css'
]

for file_path in static_files:
    full_path = os.path.join(settings.BASE_DIR, file_path)
    if os.path.exists(full_path):
        print(f"   ✅ {file_path}")
    else:
        print(f"   ⚠️  {file_path} - No encontrado")

# 8. Verificar templates
print("\n8️⃣ Verificando templates...")
template_files = [
    'templates/skin_detector/base.html',
    'templates/skin_detector/home.html',
    'templates/skin_detector/history.html'
]

for file_path in template_files:
    full_path = os.path.join(settings.BASE_DIR, file_path)
    if os.path.exists(full_path):
        print(f"   ✅ {file_path}")
    else:
        print(f"   ⚠️  {file_path} - No encontrado")

# 9. Test de predicción (opcional)
print("\n9️⃣ Test de predicción con datos dummy...")
try:
    # Crear imagen de prueba
    from PIL import Image
    import tempfile
    
    # Crear imagen RGB 224x224
    test_image = Image.new('RGB', (224, 224), color='red')
    
    # Guardar temporalmente
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        test_image.save(tmp.name)
        temp_path = tmp.name
    
    # Predecir
    result = predictor.predict(temp_path)
    
    # Limpiar
    os.unlink(temp_path)
    
    if result['success']:
        print(f"   ✅ Predicción exitosa")
        print(f"   - Clase: {result['predicted_class']}")
        print(f"   - Confianza: {result['confidence_percentage']:.2f}%")
        print(f"   - Tiempo: {result['prediction_time']:.3f}s")
    else:
        print(f"   ⚠️  Predicción falló: {result.get('error', 'Unknown')}")
        
except Exception as e:
    print(f"   ⚠️  No se pudo realizar test de predicción: {e}")

# 10. Resumen final
print("\n" + "=" * 80)
print("📊 RESUMEN DE LA VERIFICACIÓN")
print("=" * 80)

checks = {
    "Dependencias instaladas": True,
    "Modelo encontrado": True,
    "Predictor cargado": True,
    "Resumen del modelo OK": True,
    "7 clases configuradas": len(summary['classes']) == 7,
    "Información médica completa": all(c in predictor.disease_info for c in expected_classes),
}

all_passed = all(checks.values())

for check_name, passed in checks.items():
    status = "✅" if passed else "❌"
    print(f"{status} {check_name}")

print("\n" + "=" * 80)
if all_passed:
    print("🎉 ¡INTEGRACIÓN VERIFICADA Y LISTA PARA PRODUCCIÓN!")
    print("=" * 80)
    print("\n📝 Próximos pasos:")
    print("   1. Iniciar servidor: python manage.py runserver")
    print("   2. Acceder a: http://127.0.0.1:8000/")
    print("   3. Subir imagen de prueba")
    print("   4. Verificar predicciones reales")
    print("\n💡 Documentación completa en: INTEGRATION_COMPLETE.md")
    sys.exit(0)
else:
    print("⚠️  VERIFICACIÓN COMPLETADA CON ADVERTENCIAS")
    print("=" * 80)
    print("\n📝 Revisar los elementos marcados con ❌ arriba")
    sys.exit(1)
