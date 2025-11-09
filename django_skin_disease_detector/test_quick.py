"""
Test rápido del validador con dataset
"""
import os
import sys
import numpy as np
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'skin_disease_project.settings')
import django
django.setup()

from skin_detector.predictor import SkinDiseasePredictor
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

predictor = SkinDiseasePredictor()

# Probar 5 imágenes del dataset
ham_dir = r'D:\SkinAI\ai-model\datasets\ham10000\HAM10000_images_part_1'
images = [f for f in os.listdir(ham_dir) if f.endswith('.jpg')][:5]

print("\n" + "="*70)
print("PROBANDO CON IMÁGENES DEL DATASET HAM10000")
print("="*70)

accepted = 0
for img_name in images:
    img_path = os.path.join(ham_dir, img_name)
    result = predictor.predict(img_path)
    
    if result.get('success', True):
        accepted += 1
        print(f"✅ {img_name[:30]:30} - ACEPTADA - {result['predicted_class']} ({result['confidence_percentage']:.1f}%)")
    else:
        print(f"❌ {img_name[:30]:30} - RECHAZADA - {result.get('message', 'Error')}")

print(f"\n📊 Resultado: {accepted}/5 aceptadas ({accepted/5*100:.0f}%)")
if accepted >= 4:
    print("✅ Sistema funcionando CORRECTAMENTE con dataset")
else:
    print("❌ Sistema rechazando demasiadas imágenes válidas")
