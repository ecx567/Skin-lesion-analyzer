"""
Script de prueba para el nuevo Skin Validator

Prueba el validador con diferentes tipos de imágenes:
1. Imágenes del dataset HAM10000 (deben ACEPTARSE)
2. Imágenes de animales (deben RECHAZARSE)
3. Imágenes de objetos (deben RECHAZARSE)
"""

import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
import logging

# Configurar Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'skin_disease_project.settings')
import django
django.setup()

from skin_detector.skin_validator import SkinValidator

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def preprocess_image(image_path):
    """Preprocesar imagen como lo hace el predictor"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array


def test_validator():
    """Probar el validador con diferentes imágenes"""
    
    logger.info("=" * 70)
    logger.info("PROBANDO SKIN VALIDATOR")
    logger.info("=" * 70)
    
    # Inicializar validador
    validator = SkinValidator()
    logger.info(f"✅ Validator inicializado")
    logger.info(f"   Min skin: {validator.min_skin_percentage}%")
    logger.info(f"   Min confidence: {validator.min_confidence}")
    logger.info(f"   Max entropy: {validator.max_entropy}")
    logger.info("")
    
    # =====================================================
    # TEST 1: Imágenes del dataset HAM10000 (DEBEN PASAR)
    # =====================================================
    logger.info("=" * 70)
    logger.info("TEST 1: Imágenes del dataset HAM10000 (deben ACEPTARSE)")
    logger.info("=" * 70)
    
    ham_images_dir = os.path.join('..', 'ai-model', 'datasets', 'ham10000', 'HAM10000_images_part_1')
    
    if os.path.exists(ham_images_dir):
        # Probar 20 imágenes aleatorias del dataset
        all_images = [f for f in os.listdir(ham_images_dir) if f.endswith('.jpg')]
        test_images = np.random.choice(all_images, min(20, len(all_images)), replace=False)
        
        accepted = 0
        rejected = 0
        
        for img_name in test_images:
            img_path = os.path.join(ham_images_dir, img_name)
            try:
                img = preprocess_image(img_path)
                result = validator.validate(img)
                
                if result['is_valid']:
                    accepted += 1
                    status = "✅ ACEPTADA"
                else:
                    rejected += 1
                    status = "❌ RECHAZADA"
                
                logger.info(f"{status} - {img_name[:20]:20} - Score: {result['confidence_score']:5.1f} - {result['message']}")
                
            except Exception as e:
                logger.error(f"Error con {img_name}: {str(e)}")
        
        logger.info("")
        logger.info(f"📊 Resultados HAM10000:")
        logger.info(f"   ✅ Aceptadas: {accepted}/{len(test_images)} ({accepted/len(test_images)*100:.1f}%)")
        logger.info(f"   ❌ Rechazadas: {rejected}/{len(test_images)} ({rejected/len(test_images)*100:.1f}%)")
        
        if accepted < len(test_images) * 0.90:  # Si rechaza más del 10%
            logger.warning("⚠️ ALERTA: Rechazando demasiadas imágenes del dataset!")
        else:
            logger.info("✅ Resultado CORRECTO: Acepta imágenes del dataset")
    else:
        logger.warning(f"⚠️ No se encontró directorio: {ham_images_dir}")
    
    logger.info("")
    
    # =====================================================
    # TEST 2: Imagen sintética de animal (DEBE RECHAZAR)
    # =====================================================
    logger.info("=" * 70)
    logger.info("TEST 2: Imagen sintética de animal (debe RECHAZARSE)")
    logger.info("=" * 70)
    
    # Crear imagen sintética que simule un perro (marrón con textura)
    fake_dog = np.random.rand(224, 224, 3)
    # Tonos marrones (no tonos de piel)
    fake_dog[:, :, 0] = np.random.uniform(0.4, 0.6, (224, 224))  # R
    fake_dog[:, :, 1] = np.random.uniform(0.3, 0.5, (224, 224))  # G
    fake_dog[:, :, 2] = np.random.uniform(0.2, 0.4, (224, 224))  # B
    
    result = validator.validate(fake_dog.astype(np.float32))
    
    status = "❌ RECHAZADA" if not result['is_valid'] else "✅ ACEPTADA (ERROR)"
    logger.info(f"{status}")
    logger.info(f"   Score: {result['confidence_score']:.1f}/100")
    logger.info(f"   Mensaje: {result['message']}")
    
    if 'color_analysis' in result['details']:
        logger.info(f"   Piel: {result['details']['color_analysis']['skin_percentage']:.1f}%")
    
    if result['is_valid']:
        logger.error("❌ ERROR: Debería rechazar imagen de animal!")
    else:
        logger.info("✅ CORRECTO: Rechaza imagen de animal")
    
    logger.info("")
    
    # =====================================================
    # TEST 3: Imagen completamente blanca (DEBE RECHAZAR)
    # =====================================================
    logger.info("=" * 70)
    logger.info("TEST 3: Imagen completamente blanca (debe RECHAZARSE)")
    logger.info("=" * 70)
    
    white_image = np.ones((224, 224, 3), dtype=np.float32)
    result = validator.validate(white_image)
    
    status = "❌ RECHAZADA" if not result['is_valid'] else "✅ ACEPTADA (ERROR)"
    logger.info(f"{status}")
    logger.info(f"   Score: {result['confidence_score']:.1f}/100")
    logger.info(f"   Mensaje: {result['message']}")
    
    if result['is_valid']:
        logger.error("❌ ERROR: Debería rechazar imagen blanca!")
    else:
        logger.info("✅ CORRECTO: Rechaza imagen blanca")
    
    logger.info("")
    
    # =====================================================
    # TEST 4: Imagen con tonos de piel sintética (DEBE ACEPTAR)
    # =====================================================
    logger.info("=" * 70)
    logger.info("TEST 4: Imagen con tonos de piel sintética (debe ACEPTARSE)")
    logger.info("=" * 70)
    
    # Crear imagen con tonos de piel
    skin_image = np.random.rand(224, 224, 3)
    skin_image[:, :, 0] = np.random.uniform(0.7, 0.9, (224, 224))  # R
    skin_image[:, :, 1] = np.random.uniform(0.5, 0.7, (224, 224))  # G
    skin_image[:, :, 2] = np.random.uniform(0.4, 0.6, (224, 224))  # B
    
    result = validator.validate(skin_image.astype(np.float32))
    
    status = "✅ ACEPTADA" if result['is_valid'] else "❌ RECHAZADA (ERROR)"
    logger.info(f"{status}")
    logger.info(f"   Score: {result['confidence_score']:.1f}/100")
    logger.info(f"   Mensaje: {result['message']}")
    
    if 'color_analysis' in result['details']:
        logger.info(f"   Piel: {result['details']['color_analysis']['skin_percentage']:.1f}%")
    
    if not result['is_valid']:
        logger.warning("⚠️ ADVERTENCIA: Rechaza imagen con tonos de piel")
    else:
        logger.info("✅ CORRECTO: Acepta imagen con tonos de piel")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("PRUEBAS COMPLETADAS")
    logger.info("=" * 70)


if __name__ == '__main__':
    test_validator()
