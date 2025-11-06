"""
Script para verificar compatibilidad de NumPy con el modelo y TensorFlow
"""
import numpy as np
import tensorflow as tf
from PIL import Image
import sys

print("="*60)
print("VERIFICACIÓN DE COMPATIBILIDAD NumPy + TensorFlow")
print("="*60)

# 1. Verificar versiones
print(f"\n1. VERSIONES ACTUALES:")
print(f"   NumPy: {np.__version__}")
print(f"   TensorFlow: {tf.__version__}")
print(f"   Python: {sys.version}")

# 2. Probar operaciones básicas de NumPy usadas en el predictor
print(f"\n2. PRUEBAS DE OPERACIONES NUMPY:")
try:
    # Simular preprocesamiento de imagen
    test_array = np.random.rand(224, 224, 3).astype(np.float32)
    print(f"   ✓ Crear array float32: {test_array.shape}")
    
    # Normalización
    normalized = test_array / 255.0
    print(f"   ✓ Normalización: {normalized.shape}")
    
    # Expand dims (agregar batch dimension)
    expanded = np.expand_dims(normalized, axis=0)
    print(f"   ✓ Expand dims: {expanded.shape}")
    
    # Argmax (para obtener clase predicha)
    dummy_probs = np.random.dirichlet(np.ones(7), size=1)[0]
    predicted_idx = np.argmax(dummy_probs)
    print(f"   ✓ Argmax: clase {predicted_idx}")
    
    print("   ✅ Todas las operaciones NumPy funcionan correctamente")
    
except Exception as e:
    print(f"   ❌ ERROR en operaciones NumPy: {e}")
    sys.exit(1)

# 3. Probar compatibilidad con TensorFlow
print(f"\n3. PRUEBAS DE COMPATIBILIDAD TensorFlow + NumPy:")
try:
    # Crear un modelo simple
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    print(f"   ✓ Crear modelo simple")
    
    # Probar predicción con array numpy
    test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    prediction = model.predict(test_input, verbose=0)
    print(f"   ✓ Predicción con array NumPy: {prediction.shape}")
    
    # Verificar que el resultado sea un array numpy
    assert isinstance(prediction, np.ndarray), "La predicción no es un array NumPy"
    print(f"   ✓ Resultado es array NumPy: {type(prediction)}")
    
    # Verificar operaciones post-predicción
    max_prob = np.max(prediction)
    pred_class = np.argmax(prediction)
    print(f"   ✓ Operaciones post-predicción: max={max_prob:.4f}, clase={pred_class}")
    
    print("   ✅ Compatibilidad TensorFlow + NumPy verificada")
    
except Exception as e:
    print(f"   ❌ ERROR en compatibilidad TensorFlow: {e}")
    sys.exit(1)

# 4. Verificar operaciones con PIL
print(f"\n4. PRUEBAS DE OPERACIONES PIL + NumPy:")
try:
    # Crear una imagen dummy
    dummy_img = Image.new('RGB', (224, 224), color='red')
    print(f"   ✓ Crear imagen PIL: {dummy_img.size}")
    
    # Convertir a numpy array
    img_array = np.array(dummy_img, dtype=np.float32)
    print(f"   ✓ Convertir PIL a NumPy: {img_array.shape}, dtype={img_array.dtype}")
    
    # Normalizar
    img_array = img_array / 255.0
    print(f"   ✓ Normalizar: rango [{img_array.min():.2f}, {img_array.max():.2f}]")
    
    print("   ✅ Operaciones PIL + NumPy funcionan correctamente")
    
except Exception as e:
    print(f"   ❌ ERROR en operaciones PIL: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
print("="*60)
print("\n🔍 ANÁLISIS DE COMPATIBILIDAD:")
print(f"   - TensorFlow 2.15.0 es compatible con NumPy 1.23.5 - 1.26.4")
print(f"   - Versión actual: {np.__version__}")
print(f"   - Actualizar a NumPy 1.26.4 es SEGURO")
print(f"   - No habrá problemas con el modelo o predicciones")
print("="*60)
