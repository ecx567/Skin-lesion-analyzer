"""
Script de prueba REAL con el modelo de predicción

Prueba el sistema completo con imágenes reales y el modelo cargado
"""

import os
import sys
import numpy as np
from PIL import Image
import logging

# Configurar Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'skin_disease_project.settings')
import django
django.setup()

from skin_detector.predictor import SkinDiseasePredictor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def test_real_images():
    """Probar con imágenes reales del dataset"""
    
    logger.info("=" * 70)
    logger.info("PROBANDO SISTEMA COMPLETO CON MODELO REAL")
    logger.info("=" * 70)
    
    # Inicializar predictor
    predictor = SkinDiseasePredictor()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST: Imágenes del dataset HAM10000")
    logger.info("=" * 70)
    
    # Ruta a las imágenes del dataset
    ham_images_dir = os.path.join('..', 'ai-model', 'datasets', 'ham10000', 'HAM10000_images_part_1')
    
    if os.path.exists(ham_images_dir):
        # Probar 10 imágenes aleatorias
        all_images = [f for f in os.listdir(ham_images_dir) if f.endswith('.jpg')]
        test_images = np.random.choice(all_images, min(10, len(all_images)), replace=False)
        
        accepted = 0
        rejected = 0
        
        for img_name in test_images:
            img_path = os.path.join(ham_images_dir, img_name)
            try:
                logger.info(f"\n🔍 Analizando: {img_name}")
                result = predictor.predict(img_path)
                
                if result.get('success', True):
                    accepted += 1
                    logger.info(f"   ✅ ACEPTADA")
                    logger.info(f"   Predicción: {result['predicted_class']} ({result['confidence_percentage']}%)")
                    
                    if 'validation' in result:
                        logger.info(f"   Validación Score: {result['validation']['confidence_score']:.1f}/100")
                else:
                    rejected += 1
                    logger.info(f"   ❌ RECHAZADA")
                    logger.info(f"   Razón: {result.get('message', 'Desconocida')}")
                    
                    if 'validation' in result:
                        details = result['validation'].get('details', {})
                        if 'color_analysis' in details:
                            logger.info(f"   Piel: {details['color_analysis']['skin_percentage']:.1f}%")
                        if 'confidence_analysis' in details and details['confidence_analysis']:
                            logger.info(f"   Confianza: {details['confidence_analysis']['max_confidence']*100:.1f}%")
                
            except Exception as e:
                logger.error(f"   ❌ Error: {str(e)}")
        
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"📊 RESULTADOS FINALES")
        logger.info("=" * 70)
        logger.info(f"   ✅ Aceptadas: {accepted}/{len(test_images)} ({accepted/len(test_images)*100:.1f}%)")
        logger.info(f"   ❌ Rechazadas: {rejected}/{len(test_images)} ({rejected/len(test_images)*100:.1f}%)")
        
        if accepted >= len(test_images) * 0.90:
            logger.info("✅ RESULTADO CORRECTO: Sistema funcionando bien")
        else:
            logger.warning("⚠️ ALERTA: Sistema rechazando demasiadas imágenes válidas")
    
    else:
        logger.error(f"❌ No se encontró el directorio: {ham_images_dir}")


if __name__ == '__main__':
    test_real_images()
