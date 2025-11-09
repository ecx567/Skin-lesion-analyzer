"""
Script de Entrenamiento del OOD Detector

Este script entrena el OOD (Out-of-Distribution) Detector usando
el dataset HAM10000 para calcular las estadísticas necesarias que
permitirán rechazar imágenes que NO sean de lesiones cutáneas.

Uso:
    python train_ood_detector.py

Resultado:
    - Crea el archivo: models/ood_detector_stats.npz
    - Este archivo contiene las estadísticas (media, covarianza, threshold)
"""

import os
import sys
import numpy as np
import pandas as pd
from tensorflow import keras
import cv2
from pathlib import Path
import logging

# Agregar path del proyecto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from skin_detector.ood_detector import OODDetector

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_ham10000_images(csv_path, images_base_folder, sample_size=1000, max_per_class=None):
    """
    Carga un conjunto balanceado de imágenes del dataset HAM10000
    
    Args:
        csv_path: Ruta al CSV con metadata (HAM10000_metadata.csv)
        images_base_folder: Carpeta base con las imágenes (ham10000/)
        sample_size: Número total de imágenes a cargar
        max_per_class: Máximo de imágenes por clase (para balance)
    
    Returns:
        numpy.ndarray: Array de imágenes preprocesadas (n, 224, 224, 3)
    """
    logger.info(f"📂 Cargando dataset desde: {csv_path}")
    
    # Leer metadata
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Archivo no encontrado: {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"📊 Total de imágenes en metadata: {len(df)}")
    
    # Mostrar distribución de clases
    logger.info("📊 Distribución de clases en el dataset:")
    class_counts = df['dx'].value_counts()
    for class_name, count in class_counts.items():
        logger.info(f"   - {class_name}: {count} imágenes")
    
    # Balance de clases si se especifica
    if max_per_class:
        logger.info(f"⚖️ Balanceando clases (max {max_per_class} por clase)...")
        df = df.groupby('dx').apply(
            lambda x: x.sample(n=min(len(x), max_per_class), random_state=42)
        ).reset_index(drop=True)
        logger.info(f"✅ Dataset balanceado: {len(df)} imágenes")
    
    # Tomar muestra si es necesario
    if sample_size and sample_size < len(df):
        logger.info(f"🎲 Tomando muestra de {sample_size} imágenes...")
        df = df.sample(n=sample_size, random_state=42)
    
    # Cargar imágenes
    images = []
    failed = 0
    
    logger.info(f"🔄 Cargando y preprocesando {len(df)} imágenes...")
    
    for idx, row in df.iterrows():
        image_id = row['image_id']
        
        # Buscar imagen en las carpetas (part_1 o part_2)
        image_path = None
        for folder in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
            potential_path = os.path.join(images_base_folder, folder, f"{image_id}.jpg")
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if not image_path:
            failed += 1
            continue
        
        try:
            # Cargar y preprocesar imagen
            img = cv2.imread(image_path)
            if img is None:
                failed += 1
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0  # Normalizar a [0, 1]
            images.append(img)
            
            # Mostrar progreso cada 100 imágenes
            if len(images) % 100 == 0:
                logger.info(f"   ✓ {len(images)}/{len(df)} imágenes procesadas")
                
        except Exception as e:
            logger.warning(f"⚠️ Error cargando {image_id}: {e}")
            failed += 1
    
    if failed > 0:
        logger.warning(f"⚠️ {failed} imágenes fallaron al cargar")
    
    images_array = np.array(images)
    logger.info(f"✅ {len(images)} imágenes cargadas exitosamente")
    logger.info(f"📐 Shape final: {images_array.shape}")
    
    return images_array


def main():
    """
    Función principal de entrenamiento
    """
    print("=" * 70)
    print("🚀 ENTRENAMIENTO DEL OOD DETECTOR - SKINAI")
    print("=" * 70)
    print()
    
    # Configuración
    MODEL_PATH = 'models/improved_balanced_7class_model.h5'
    CSV_PATH = '../ai-model/datasets/ham10000/HAM10000_metadata.csv'
    IMAGES_FOLDER = '../ai-model/datasets/ham10000'
    OUTPUT_PATH = 'models/ood_detector_stats.npz'
    
    # Parámetros de entrenamiento ÓPTIMOS - Mayor cobertura del dataset
    # Máximo posible sin causar error de memoria
    SAMPLE_SIZE = 4000  # 4000 imágenes (máximo sin error de RAM)
    MAX_PER_CLASS = 700  # 700 por clase (balance pero con más datos)
    PERCENTILE = 99  # CRÍTICO: 99% para minimizar falsos positivos en lesiones
    FILTER_OUTLIERS = False  # DESACTIVADO: Todas las imágenes son válidas
    OUTLIER_THRESHOLD_PERCENTILE = 98  # No se usa
    LAYER_NAME = 'dense'  # Capa para extraer características
    
    logger.info("⚙️ Configuración:")
    logger.info(f"   - Modelo: {MODEL_PATH}")
    logger.info(f"   - Dataset CSV: {CSV_PATH}")
    logger.info(f"   - Imágenes: {IMAGES_FOLDER}")
    logger.info(f"   - Salida: {OUTPUT_PATH}")
    logger.info(f"   - Sample size: {SAMPLE_SIZE}")
    logger.info(f"   - Max por clase: {MAX_PER_CLASS}")
    logger.info(f"   - Percentil: {PERCENTILE}")
    logger.info(f"   - Filtrar outliers: {FILTER_OUTLIERS}")
    if FILTER_OUTLIERS:
        logger.info(f"   - Outlier threshold: {OUTLIER_THRESHOLD_PERCENTILE}%")
    logger.info(f"   - Capa: {LAYER_NAME}")
    print()
    
    # Verificar que existan los archivos
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Modelo no encontrado: {MODEL_PATH}")
        logger.error("   Asegúrate de que el modelo esté en la carpeta 'models/'")
        return
    
    if not os.path.exists(CSV_PATH):
        logger.error(f"❌ CSV no encontrado: {CSV_PATH}")
        logger.error("   Asegúrate de que el dataset HAM10000 esté en '../ai-model/datasets/ham10000/'")
        return
    
    try:
        # 1. Cargar modelo principal
        logger.info("=" * 70)
        logger.info("PASO 1: CARGANDO MODELO DE CLASIFICACIÓN")
        logger.info("=" * 70)
        
        # Cargar modelo SIN compilar (evita error con Focal Loss personalizada)
        logger.info("🔄 Cargando modelo (sin compilar)...")
        model = keras.models.load_model(MODEL_PATH, compile=False)
        logger.info("✅ Modelo cargado exitosamente")
        logger.info(f"📊 Arquitectura: {len(model.layers)} capas")
        
        # NO necesitamos recompilar para OOD Detection
        # (solo usamos las características, no las predicciones)
        logger.info("ℹ️ Modelo cargado sin compilar (solo necesitamos features)")
        
        # Llamada dummy para construir el modelo completamente
        logger.info("🔧 Construyendo modelo con llamada dummy...")
        import numpy as np
        dummy_input = np.zeros((1, 224, 224, 3))
        _ = model(dummy_input, training=False)
        logger.info("✅ Modelo construido exitosamente")
        print()
        
        # Mostrar capas disponibles
        logger.info("📋 Capas disponibles en el modelo:")
        for i, layer in enumerate(model.layers):
            logger.info(f"   {i}: {layer.name} ({layer.__class__.__name__})")
        print()
        
        # 2. Cargar imágenes de entrenamiento del HAM10000
        logger.info("=" * 70)
        logger.info("PASO 2: CARGANDO IMÁGENES DEL DATASET HAM10000")
        logger.info("=" * 70)
        train_images = load_ham10000_images(
            csv_path=CSV_PATH,
            images_base_folder=IMAGES_FOLDER,
            sample_size=SAMPLE_SIZE,
            max_per_class=MAX_PER_CLASS
        )
        print()
        
        if len(train_images) == 0:
            logger.error("❌ No se pudieron cargar imágenes. Verifica las rutas.")
            return
        
        # 3. Crear y entrenar OOD Detector
        logger.info("=" * 70)
        logger.info("PASO 3: CREANDO OOD DETECTOR")
        logger.info("=" * 70)
        
        # IMPORTANTE: Hacer una llamada dummy al modelo para construir el grafo
        # Esto es necesario en Keras 3.x cuando se carga sin compilar
        logger.info("🔧 Inicializando modelo con datos dummy...")
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        _ = model.predict(dummy_input, verbose=0)
        logger.info("✅ Modelo inicializado correctamente")
        
        ood_detector = OODDetector(model, layer_name=LAYER_NAME)
        logger.info("✅ OOD Detector creado")
        print()
        
        logger.info("=" * 70)
        logger.info("PASO 4: ENTRENANDO OOD DETECTOR")
        logger.info("=" * 70)
        logger.info("⏳ Esto puede tardar 2-5 minutos dependiendo de tu hardware...")
        print()
        
        # PASO 4.1: Entrenar preliminar para detectar outliers
        if FILTER_OUTLIERS:
            logger.info("🔍 PASO 4.1: Entrenamiento preliminar para detectar outliers...")
            logger.info(f"   Entrenando con percentil {OUTLIER_THRESHOLD_PERCENTILE} temporal...")
            
            # Entrenar con percentil alto para detectar outliers
            prelim_stats = ood_detector.fit(train_images, percentile=OUTLIER_THRESHOLD_PERCENTILE)
            
            # Extraer features de todas las imágenes
            logger.info("   Extrayendo características para análisis...")
            features = ood_detector.feature_extractor.predict(train_images, verbose=0, batch_size=32)
            
            # Calcular distancias
            logger.info("   Calculando distancias Mahalanobis...")
            from scipy.spatial.distance import mahalanobis
            distances = []
            for feature in features:
                dist = mahalanobis(feature, ood_detector.mean, ood_detector.inv_cov)
                distances.append(dist)
            distances = np.array(distances)
            
            # Filtrar outliers (eliminar 2% superior)
            outlier_threshold = np.percentile(distances, OUTLIER_THRESHOLD_PERCENTILE)
            mask = distances <= outlier_threshold
            
            n_outliers = np.sum(~mask)
            logger.info(f"   📊 Outliers detectados: {n_outliers}/{len(train_images)} ({n_outliers/len(train_images)*100:.1f}%)")
            logger.info(f"   📊 Distancia outlier threshold: {outlier_threshold:.2f}")
            
            # Filtrar imágenes
            train_images_filtered = train_images[mask]
            logger.info(f"   ✅ Dataset filtrado: {len(train_images_filtered)} imágenes (eliminados {n_outliers} outliers)")
            print()
            
            # PASO 4.2: Re-entrenar con dataset limpio
            logger.info("🔄 PASO 4.2: Re-entrenamiento con dataset limpio...")
            logger.info(f"   Entrenando con percentil {PERCENTILE} final...")
            stats = ood_detector.fit(train_images_filtered, percentile=PERCENTILE)
        else:
            # Entrenar normalmente sin filtrado
            stats = ood_detector.fit(train_images, percentile=PERCENTILE)
        
        print()
        
        logger.info("=" * 70)
        logger.info("ESTADÍSTICAS DEL ENTRENAMIENTO")
        logger.info("=" * 70)
        logger.info(f"✅ Muestras procesadas: {stats['n_samples']}")
        logger.info(f"✅ Dimensión de características: {stats['feature_dim']}")
        logger.info(f"✅ Threshold calculado: {stats['threshold']:.2f}")
        logger.info(f"✅ Distancia promedio: {stats['mean_distance']:.2f} ± {stats['std_distance']:.2f}")
        logger.info(f"✅ Rango de distancias: [{stats['min_distance']:.2f}, {stats['max_distance']:.2f}]")
        print()
        
        # 4. Guardar detector
        logger.info("=" * 70)
        logger.info("PASO 5: GUARDANDO OOD DETECTOR")
        logger.info("=" * 70)
        ood_detector.save(OUTPUT_PATH)
        print()
        
        # 5. VALIDACIÓN: Probar con imágenes del training set
        logger.info("=" * 70)
        logger.info("PASO 6: VALIDACIÓN - PROBANDO CON IMÁGENES DE ENTRENAMIENTO")
        logger.info("=" * 70)
        logger.info("🧪 Probando que NO rechace imágenes válidas del training set...")
        
        # Tomar muestra aleatoria de 50 imágenes del training set
        test_indices = np.random.choice(len(train_images), size=min(50, len(train_images)), replace=False)
        test_images = train_images[test_indices]
        
        rejected = 0
        accepted = 0
        distances = []
        
        for i, img in enumerate(test_images):
            result = ood_detector.predict(np.expand_dims(img, axis=0))
            distances.append(result['distance'])
            
            if result['is_valid']:
                accepted += 1
            else:
                rejected += 1
                logger.warning(f"   ⚠️ Imagen {i+1} RECHAZADA (distancia: {result['distance']:.2f})")
        
        acceptance_rate = (accepted / len(test_images)) * 100
        logger.info(f"\n📊 Resultados de validación:")
        logger.info(f"   ✅ Aceptadas: {accepted}/{len(test_images)} ({acceptance_rate:.1f}%)")
        logger.info(f"   ❌ Rechazadas: {rejected}/{len(test_images)} ({100-acceptance_rate:.1f}%)")
        logger.info(f"   📏 Distancia promedio en test: {np.mean(distances):.2f}")
        logger.info(f"   🎯 Threshold actual: {stats['threshold']:.2f}")
        
        if acceptance_rate < 95:
            logger.warning(f"\n⚠️ ADVERTENCIA: Tasa de aceptación baja ({acceptance_rate:.1f}%)")
            logger.warning("   El threshold podría ser muy restrictivo.")
            logger.warning("   Considera aumentar el PERCENTILE a 99.5 o entrenar con más imágenes.")
        else:
            logger.info(f"\n✅ Validación exitosa: {acceptance_rate:.1f}% de imágenes válidas aceptadas")
        print()
        
        # 6. Verificar que se guardó correctamente
        logger.info("=" * 70)
        logger.info("PASO 7: VERIFICACIÓN FINAL")
        logger.info("=" * 70)
        if os.path.exists(OUTPUT_PATH):
            file_size = os.path.getsize(OUTPUT_PATH) / 1024  # KB
            logger.info(f"✅ Archivo guardado: {OUTPUT_PATH} ({file_size:.1f} KB)")
        else:
            logger.error(f"❌ Error: no se pudo guardar el archivo")
            return
        
        print()
        logger.info("=" * 70)
        logger.info("🎉 ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
        logger.info("=" * 70)
        print()
        logger.info("📝 Próximos pasos:")
        logger.info("   1. El archivo 'ood_detector_stats.npz' ya está listo")
        logger.info("   2. El sistema Django lo cargará automáticamente")
        logger.info("   3. Ahora las imágenes no-cutáneas serán rechazadas")
        print()
        logger.info("🧪 Para probar el OOD Detector:")
        logger.info("   - Sube una imagen de una lesión cutánea → debe aceptarse ✅")
        logger.info("   - Sube una imagen de un paisaje → debe rechazarse ❌")
        logger.info("   - Sube una imagen de un animal → debe rechazarse ❌")
        print()
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Entrenamiento interrumpido por el usuario")
        return
        
    except Exception as e:
        logger.error(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
