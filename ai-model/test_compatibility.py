#!/usr/bin/env python3
"""
Script para verificar compatibilidad entre TensorFlow, Keras y otros paquetes
"""
import sys
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🔍 VERIFICACIÓN DE COMPATIBILIDAD DE PAQUETES")
print("=" * 70)

# 1. Versiones instaladas
print("\n📦 VERSIONES INSTALADAS:")
print("-" * 70)

try:
    import tensorflow as tf
    print(f"✅ TensorFlow: {tf.__version__}")
    print(f"   - Keras integrado: {tf.keras.__version__}")
except ImportError as e:
    print(f"❌ TensorFlow: No instalado - {e}")
    sys.exit(1)

try:
    import keras
    print(f"✅ Keras (standalone): {keras.__version__}")
except ImportError as e:
    print(f"❌ Keras: No instalado - {e}")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy: No instalado - {e}")

try:
    import pandas as pd
    print(f"✅ Pandas: {pd.__version__}")
except ImportError as e:
    print(f"⚠️  Pandas: No instalado - {e}")

try:
    import sklearn
    print(f"✅ scikit-learn: {sklearn.__version__}")
except ImportError as e:
    print(f"⚠️  scikit-learn: No instalado - {e}")

try:
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"⚠️  OpenCV: No instalado - {e}")

try:
    import matplotlib
    print(f"✅ Matplotlib: {matplotlib.__version__}")
except ImportError as e:
    print(f"⚠️  Matplotlib: No instalado - {e}")

try:
    import h5py
    print(f"✅ h5py: {h5py.__version__}")
except ImportError as e:
    print(f"⚠️  h5py: No instalado - {e}")

# 2. Compatibilidad Keras-TensorFlow
print("\n🔗 COMPATIBILIDAD KERAS-TENSORFLOW:")
print("-" * 70)

keras_standalone_version = keras.__version__
tf_keras_version = tf.keras.__version__

if keras_standalone_version == tf_keras_version:
    print(f"✅ COMPATIBLES: Keras {keras_standalone_version} == TF.Keras {tf_keras_version}")
else:
    print(f"⚠️  ADVERTENCIA: Keras {keras_standalone_version} != TF.Keras {tf_keras_version}")
    print("   Esto podría causar problemas de compatibilidad.")

# 3. Prueba de funcionalidad básica
print("\n🧪 PRUEBAS DE FUNCIONALIDAD:")
print("-" * 70)

try:
    print("   Probando creación de modelo simple...")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    
    model = Sequential([
        Dense(10, activation='relu', input_shape=(5,)),
        Dense(3, activation='softmax')
    ])
    print("   ✅ Modelo Sequential creado correctamente")
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("   ✅ Modelo compilado correctamente")
    
    # Prueba con datos dummy
    X_dummy = np.random.rand(10, 5)
    y_dummy = np.random.randint(0, 3, size=(10,))
    y_dummy = tf.keras.utils.to_categorical(y_dummy, 3)
    
    model.fit(X_dummy, y_dummy, epochs=1, verbose=0)
    print("   ✅ Entrenamiento de prueba exitoso")
    
    predictions = model.predict(X_dummy, verbose=0)
    print("   ✅ Predicción de prueba exitosa")
    
except Exception as e:
    print(f"   ❌ ERROR: {e}")

# 4. Verificar compatibilidad con Keras 3.x
print("\n📋 CARACTERÍSTICAS DE KERAS 3.x:")
print("-" * 70)

keras_major_version = int(keras.__version__.split('.')[0])

if keras_major_version >= 3:
    print("✅ Usando Keras 3.x (Multi-backend)")
    print("   - Soporta: TensorFlow, JAX, PyTorch")
    print("   ⚠️  NOTA: Keras 3.x tiene cambios importantes respecto a Keras 2.x")
    print("   ⚠️  Los modelos .h5 de Keras 2.x pueden tener problemas de compatibilidad")
    print("\n   📌 RECOMENDACIONES:")
    print("   1. Si tienes un modelo .h5 antiguo (Keras 2.x):")
    print("      - Necesitarás re-entrenar con Keras 3.x")
    print("      - O usar TensorFlow 2.15.x con Keras 2.15.x")
    print("   2. Para nuevos modelos: Keras 3.x es compatible")
else:
    print(f"✅ Usando Keras {keras_major_version}.x (Tradicional)")

# 5. Compatibilidad NumPy
print("\n🔢 COMPATIBILIDAD NUMPY:")
print("-" * 70)

numpy_version = tuple(map(int, np.__version__.split('.')))
tf_version = tuple(map(int, tf.__version__.split('.')[:2]))

if tf_version >= (2, 20):
    if numpy_version >= (1, 23) and numpy_version <= (2, 3):
        print(f"✅ NumPy {np.__version__} compatible con TensorFlow {tf.__version__}")
    else:
        print(f"⚠️  NumPy {np.__version__} podría tener problemas con TensorFlow {tf.__version__}")
        print("   Rango recomendado: NumPy 1.23.x - 2.2.x")

# 6. Resumen final
print("\n" + "=" * 70)
print("📊 RESUMEN DE COMPATIBILIDAD")
print("=" * 70)

issues = []

# Keras 3.x con modelos antiguos
if keras_major_version >= 3:
    issues.append("⚠️  Keras 3.x: Modelos .h5 antiguos (Keras 2.x) NO son compatibles")
    issues.append("   Solución: Re-entrenar el modelo con Keras 3.x")

# NumPy muy nuevo
if numpy_version[0] >= 2 and numpy_version[1] >= 3:
    issues.append("⚠️  NumPy 2.3+: Versión muy reciente, podría tener problemas")

if not issues:
    print("✅ TODOS LOS PAQUETES SON COMPATIBLES")
    print("✅ El entorno está listo para entrenar modelos nuevos")
else:
    print("⚠️  SE ENCONTRARON POSIBLES PROBLEMAS DE COMPATIBILIDAD:")
    for issue in issues:
        print(f"   {issue}")

print("\n💡 CONCLUSIÓN:")
if keras_major_version >= 3:
    print("   - Para NUEVOS modelos: ✅ Todo listo para entrenar")
    print("   - Para modelos ANTIGUOS (.h5 Keras 2.x): ❌ Necesitas re-entrenar")
else:
    print("   - ✅ Completamente compatible para entrenar y cargar modelos")

print("\n" + "=" * 70)
