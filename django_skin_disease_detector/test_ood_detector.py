"""
Test del OOD Detector - Verificar que funciona correctamente
"""
import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf

# Agregar el directorio del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configurar Django ANTES de importar módulos que lo usan
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'skin_disease_project.settings')
import django
django.setup()

from skin_detector.ood_detector import OODDetector
from skin_detector.predictor import SkinDiseasePredictor

def preprocess_image(image_path):
    """Preprocesar imagen para el modelo"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def test_ood_detector():
    print("=" * 70)
    print("🧪 TEST DEL OOD DETECTOR")
    print("=" * 70)
    
    # 1. Cargar modelo
    print("\n1️⃣ Cargando modelo...")
    model_path = 'models/improved_balanced_7class_model.h5'
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"   ✅ Modelo cargado: {len(model.layers)} capas")
    
    # 2. Cargar OOD Detector
    print("\n2️⃣ Cargando OOD Detector...")
    ood_detector = OODDetector(model, layer_name='dense')
    stats_path = 'models/ood_detector_stats.npz'
    ood_detector.load(stats_path)
    print(f"   ✅ OOD Detector cargado desde: {stats_path}")
    print(f"   📊 Threshold: {ood_detector.threshold:.2f}")
    
    # 3. Probar con imagen del dataset HAM10000 (debe ser IN-distribution)
    print("\n3️⃣ Probando con imagen IN-distribution (lesión cutánea)...")
    test_image = '../ai-model/datasets/ham10000/HAM10000_images_part_1/ISIC_0024306.jpg'
    
    if os.path.exists(test_image):
        img_array = preprocess_image(test_image)
        result = ood_detector.predict(img_array)
        
        print(f"   📷 Imagen: {os.path.basename(test_image)}")
        print(f"   📏 Distancia Mahalanobis: {result['distance']:.2f}")
        print(f"   🎯 Threshold: {result['threshold']:.2f}")
        print(f"   📊 Confianza: {result['confidence']:.2%}")
        print(f"   📊 Ratio: {result['ratio']:.2f}")
        print(f"   ✅ Resultado: {result['severity'].upper()}")
        print(f"   💬 Mensaje: {result['message']}")
        
        if not result['is_valid']:
            print("   ⚠️ ALERTA: Imagen de lesión cutánea fue rechazada!")
        else:
            print("   ✅ CORRECTO: Imagen de lesión cutánea fue aceptada")
    else:
        print(f"   ⚠️ Imagen de prueba no encontrada: {test_image}")
    
    # 4. Verificar integración con SkinDiseasePredictor
    print("\n4️⃣ Probando integración con SkinDiseasePredictor...")
    try:
        predictor = SkinDiseasePredictor()
        
        if predictor.ood_enabled:
            print("   ✅ OOD Detector integrado correctamente en SkinDiseasePredictor")
            print(f"   📊 Threshold: {predictor.ood_detector.threshold:.2f}")
        else:
            print("   ⚠️ OOD Detector NO está habilitado en SkinDiseasePredictor")
    except Exception as e:
        print(f"   ❌ Error al cargar predictor: {e}")
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETADO")
    print("=" * 70)

if __name__ == '__main__':
    test_ood_detector()
