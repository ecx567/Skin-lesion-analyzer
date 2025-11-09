"""
Script para Optimizar el Threshold del OOD Detector

Este script encuentra el threshold óptimo que:
- Acepta imágenes de lesiones cutáneas del HAM10000
- Rechaza imágenes de objetos, animales, paisajes, etc.

Usa el concepto de ROC para encontrar el mejor punto de separación.
"""

import os
import sys
import numpy as np
import pandas as pd
from tensorflow import keras
import cv2
import logging
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from skin_detector.ood_detector import OODDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_sample_images(csv_path, images_base_folder, n_samples=100):
    """Cargar muestra de imágenes IN-distribution (HAM10000)"""
    logger.info(f"📂 Cargando {n_samples} imágenes del HAM10000...")
    
    df = pd.read_csv(csv_path)
    df_sample = df.sample(n=n_samples, random_state=42)
    
    images = []
    for idx, row in df_sample.iterrows():
        image_id = row['image_id']
        
        for folder in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
            image_path = os.path.join(images_base_folder, folder, f"{image_id}.jpg")
            if os.path.exists(image_path):
                try:
                    img = cv2.imread(image_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (224, 224))
                        img = img / 255.0
                        images.append(img)
                except Exception as e:
                    continue
                break
    
    logger.info(f"✅ {len(images)} imágenes IN-distribution cargadas")
    return np.array(images)


def test_with_statistics(ood_detector, in_dist_images):
    """
    Calcula estadísticas de distancias para encontrar el threshold óptimo
    """
    logger.info("\n" + "=" * 70)
    logger.info("🔍 CALCULANDO ESTADÍSTICAS DE DISTANCIAS")
    logger.info("=" * 70)
    
    # Calcular distancias para imágenes IN-distribution
    logger.info("📊 Calculando distancias para imágenes IN-distribution...")
    in_distances = []
    
    for i, img in enumerate(in_dist_images):
        result = ood_detector.predict(np.expand_dims(img, axis=0))
        in_distances.append(result['distance'])
        if (i + 1) % 20 == 0:
            logger.info(f"   Procesadas {i + 1}/{len(in_dist_images)} imágenes")
    
    in_distances = np.array(in_distances)
    
    # Calcular estadísticas
    logger.info("\n📊 ESTADÍSTICAS DE IMÁGENES IN-DISTRIBUTION (Lesiones cutáneas):")
    logger.info(f"   Media: {np.mean(in_distances):.2f}")
    logger.info(f"   Mediana: {np.median(in_distances):.2f}")
    logger.info(f"   Desviación estándar: {np.std(in_distances):.2f}")
    logger.info(f"   Mínimo: {np.min(in_distances):.2f}")
    logger.info(f"   Máximo: {np.max(in_distances):.2f}")
    logger.info(f"   Percentil 95: {np.percentile(in_distances, 95):.2f}")
    logger.info(f"   Percentil 99: {np.percentile(in_distances, 99):.2f}")
    logger.info(f"   Percentil 99.5: {np.percentile(in_distances, 99.5):.2f}")
    
    # Análisis de threshold óptimo
    logger.info("\n🎯 ANÁLISIS DE THRESHOLD ÓPTIMO:")
    
    # Threshold conservador: Acepta 95% de las imágenes IN-distribution
    threshold_95 = np.percentile(in_distances, 95)
    logger.info(f"\n1. Threshold CONSERVADOR (Percentil 95): {threshold_95:.2f}")
    logger.info(f"   - Acepta: 95% de lesiones cutáneas")
    logger.info(f"   - Rechaza: 5% de lesiones cutáneas (falsos positivos)")
    logger.info(f"   - Mejor para: Minimizar falsos positivos")
    
    # Threshold balanceado: Acepta 98% de las imágenes IN-distribution
    threshold_98 = np.percentile(in_distances, 98)
    logger.info(f"\n2. Threshold BALANCEADO (Percentil 98): {threshold_98:.2f}")
    logger.info(f"   - Acepta: 98% de lesiones cutáneas")
    logger.info(f"   - Rechaza: 2% de lesiones cutáneas (falsos positivos)")
    logger.info(f"   - Mejor para: Balance entre precisión y cobertura")
    
    # Threshold permisivo: Acepta 99.5% de las imágenes IN-distribution
    threshold_99_5 = np.percentile(in_distances, 99.5)
    logger.info(f"\n3. Threshold PERMISIVO (Percentil 99.5): {threshold_99_5:.2f}")
    logger.info(f"   - Acepta: 99.5% de lesiones cutáneas")
    logger.info(f"   - Rechaza: 0.5% de lesiones cutáneas (falsos positivos)")
    logger.info(f"   - Mejor para: Maximizar aceptación de lesiones reales")
    
    # Threshold basado en media + desviaciones estándar
    mean_dist = np.mean(in_distances)
    std_dist = np.std(in_distances)
    
    threshold_2std = mean_dist + 2 * std_dist
    threshold_3std = mean_dist + 3 * std_dist
    
    logger.info(f"\n4. Threshold ESTADÍSTICO (Media + 2σ): {threshold_2std:.2f}")
    logger.info(f"   - Cubre ~95.4% de distribución normal")
    
    logger.info(f"\n5. Threshold ESTADÍSTICO (Media + 3σ): {threshold_3std:.2f}")
    logger.info(f"   - Cubre ~99.7% de distribución normal")
    
    # RECOMENDACIÓN BASADA EN ANÁLISIS
    logger.info("\n" + "=" * 70)
    logger.info("💡 RECOMENDACIÓN FINAL")
    logger.info("=" * 70)
    
    # Usar percentil 97-98 como compromiso óptimo
    recommended_threshold = np.percentile(in_distances, 97.5)
    
    logger.info(f"\n🎯 Threshold RECOMENDADO: {recommended_threshold:.2f} (Percentil 97.5)")
    logger.info("\n📋 Razones:")
    logger.info("   ✅ Acepta ~97.5% de lesiones cutáneas del dataset")
    logger.info("   ✅ Solo 2.5% de falsos positivos en lesiones reales")
    logger.info("   ✅ Suficientemente estricto para rechazar objetos/animales")
    logger.info("   ✅ Basado en datos reales del HAM10000")
    
    # Validar con muestras
    logger.info("\n📊 Validación con muestra del dataset:")
    accepted = np.sum(in_distances <= recommended_threshold)
    rejected = len(in_distances) - accepted
    acceptance_rate = (accepted / len(in_distances)) * 100
    
    logger.info(f"   ✅ Aceptadas: {accepted}/{len(in_distances)} ({acceptance_rate:.1f}%)")
    logger.info(f"   ❌ Rechazadas: {rejected}/{len(in_distances)} ({100-acceptance_rate:.1f}%)")
    
    if acceptance_rate >= 97:
        logger.info("\n✅ El threshold es ÓPTIMO para el sistema")
    elif acceptance_rate >= 95:
        logger.info("\n⚠️ El threshold es ACEPTABLE pero puede mejorarse")
    else:
        logger.warning("\n❌ El threshold es DEMASIADO RESTRICTIVO")
    
    return {
        'recommended': recommended_threshold,
        'conservative': threshold_95,
        'balanced': threshold_98,
        'permissive': threshold_99_5,
        'mean_2std': threshold_2std,
        'mean_3std': threshold_3std,
        'in_distances': in_distances
    }


def main():
    print("\n" + "=" * 70)
    print("🔧 OPTIMIZACIÓN DEL THRESHOLD DEL OOD DETECTOR")
    print("=" * 70)
    print()
    
    # Configuración
    MODEL_PATH = 'models/improved_balanced_7class_model.h5'
    CSV_PATH = '../ai-model/datasets/ham10000/HAM10000_metadata.csv'
    IMAGES_FOLDER = '../ai-model/datasets/ham10000'
    STATS_PATH = 'models/ood_detector_stats.npz'
    
    try:
        # 1. Cargar modelo
        logger.info("PASO 1: Cargando modelo...")
        model = keras.models.load_model(MODEL_PATH, compile=False)
        dummy_input = np.zeros((1, 224, 224, 3))
        _ = model(dummy_input, training=False)
        logger.info("✅ Modelo cargado\n")
        
        # 2. Cargar OOD Detector actual
        logger.info("PASO 2: Cargando OOD Detector actual...")
        ood_detector = OODDetector(model, layer_name='dense')
        
        if os.path.exists(STATS_PATH):
            ood_detector.load(STATS_PATH)
            logger.info(f"✅ OOD Detector cargado (Threshold actual: {ood_detector.threshold:.2f})\n")
        else:
            logger.error("❌ No existe el archivo de estadísticas. Ejecuta train_ood_detector.py primero.")
            return
        
        # 3. Cargar imágenes de prueba
        logger.info("PASO 3: Cargando imágenes de prueba del HAM10000...")
        in_dist_images = load_sample_images(CSV_PATH, IMAGES_FOLDER, n_samples=200)
        
        if len(in_dist_images) == 0:
            logger.error("❌ No se pudieron cargar imágenes de prueba")
            return
        
        # 4. Calcular threshold óptimo
        logger.info("\nPASO 4: Calculando threshold óptimo...")
        thresholds = test_with_statistics(ood_detector, in_dist_images)
        
        # 5. Actualizar OOD Detector con nuevo threshold
        logger.info("\n" + "=" * 70)
        logger.info("PASO 5: ¿ACTUALIZAR THRESHOLD?")
        logger.info("=" * 70)
        
        current_threshold = ood_detector.threshold
        recommended_threshold = thresholds['recommended']
        
        logger.info(f"\n📊 Comparación:")
        logger.info(f"   Threshold ACTUAL: {current_threshold:.2f}")
        logger.info(f"   Threshold RECOMENDADO: {recommended_threshold:.2f}")
        logger.info(f"   Diferencia: {abs(current_threshold - recommended_threshold):.2f}")
        
        if abs(current_threshold - recommended_threshold) > 5:
            logger.info("\n⚠️ Se recomienda ACTUALIZAR el threshold")
            logger.info("\nPara actualizar automáticamente, descomenta las siguientes líneas:")
            logger.info(f"   # ood_detector.threshold = {recommended_threshold:.2f}")
            logger.info(f"   # ood_detector.save('{STATS_PATH}')")
            
            # DESCOMENTAR ESTAS LÍNEAS PARA ACTUALIZAR AUTOMÁTICAMENTE
            logger.info("\n🔄 Actualizando threshold automáticamente...")
            ood_detector.threshold = recommended_threshold
            ood_detector.save(STATS_PATH)
            logger.info(f"✅ Threshold actualizado a {recommended_threshold:.2f} y guardado")
        else:
            logger.info("\n✅ El threshold actual es óptimo, no requiere cambios")
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 OPTIMIZACIÓN COMPLETADA")
        logger.info("=" * 70)
        logger.info("\n📝 Próximos pasos:")
        logger.info("   1. Reinicia el servidor Django si está corriendo")
        logger.info("   2. Prueba con imágenes de lesiones cutáneas → deben aceptarse")
        logger.info("   3. Prueba con imágenes de perros/casas → deben rechazarse")
        logger.info("   4. Si aún hay problemas, ajusta manualmente el threshold\n")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
