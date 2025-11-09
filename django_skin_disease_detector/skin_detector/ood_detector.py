"""
OOD (Out-of-Distribution) Detector usando Mahalanobis Distance

Este módulo detecta imágenes que NO son de lesiones cutáneas comparando
con la distribución de características del dataset HAM10000.

Basado en el paper: "A Simple Unified Framework for Detecting Out-of-Distribution 
Samples and Adversarial Attacks" (NeurIPS 2018)
https://arxiv.org/abs/1807.03888
"""

import numpy as np
import tensorflow as tf
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance
import os
import logging

logger = logging.getLogger(__name__)


class OODDetector:
    """
    Detector de Out-of-Distribution (OOD) usando Distancia de Mahalanobis
    
    Detecta imágenes que NO son de lesiones cutáneas comparando con
    la distribución de características del dataset HAM10000.
    
    Attributes:
        model: Modelo de Keras entrenado
        layer_name: Nombre de la capa para extraer características
        mean: Vector de medias de las características (training)
        cov: Matriz de covarianza de las características (training)
        inv_cov: Matriz de covarianza inversa
        threshold: Umbral de distancia para clasificar como OOD
    """
    
    def __init__(self, model, layer_name='dense'):
        """
        Inicializa el OOD Detector
        
        Args:
            model: Modelo de Keras entrenado (SkinDiseasePredictor model)
            layer_name: Nombre de la capa para extraer características.
                       Por defecto 'dense' (última capa densa antes de softmax)
        """
        self.model = model
        self.layer_name = layer_name
        
        # Asegurarse de que el modelo esté construido
        # (necesario para Keras 3.x cuando se carga sin compilar)
        try:
            # Intentar acceder a model.input para verificar si está construido
            _ = model.input
            model_input = model.input
        except (AttributeError, ValueError):
            # Si no está construido, hacer una llamada dummy y usar layers[0].input
            logger.info("🔧 Modelo no construido. Inicializando con datos dummy...")
            dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
            _ = model(dummy_input, training=False)
            logger.info("✅ Modelo construido exitosamente")
            # En Keras 3.x, usar la entrada de la primera capa
            model_input = model.layers[0].input
        
        # Crear modelo para extraer características de una capa intermedia
        try:
            layer = self.model.get_layer(layer_name)
            self.feature_extractor = tf.keras.Model(
                inputs=model_input,
                outputs=layer.output
            )
            logger.info(f"✅ Feature extractor creado con capa: {layer_name}")
        except ValueError:
            # Si no existe la capa 'dense', usar la penúltima capa
            logger.warning(f"⚠️ Capa '{layer_name}' no encontrada. Usando penúltima capa.")
            self.feature_extractor = tf.keras.Model(
                inputs=model_input,
                outputs=model.layers[-2].output
            )
        
        # Estadísticas de la distribución (se calculan con fit())
        self.mean = None
        self.cov = None
        self.inv_cov = None
        self.threshold = None
        self.is_fitted = False
        
    def fit(self, train_images, percentile=95):
        """
        Calcula estadísticas (media y covarianza) de las imágenes de entrenamiento
        
        Este método debe ejecutarse UNA VEZ con imágenes del dataset HAM10000
        para calcular la distribución de características de lesiones cutáneas válidas.
        
        Args:
            train_images: Array de imágenes del dataset HAM10000 (lesiones cutáneas)
                         Shape: (n_samples, 224, 224, 3), valores [0, 1]
            percentile: Percentil para calcular el threshold (default: 95)
                       95 significa que el 5% de las imágenes de entrenamiento
                       serán consideradas como "límite" del threshold
        
        Returns:
            dict: Estadísticas del entrenamiento
        """
        logger.info("🔍 Iniciando entrenamiento del OOD Detector...")
        logger.info(f"📊 Procesando {len(train_images)} imágenes de entrenamiento")
        
        # 1. Extraer características de todas las imágenes de entrenamiento
        logger.info("🔄 Extrayendo características...")
        features = self.feature_extractor.predict(train_images, verbose=0, batch_size=32)
        
        # 2. Aplanar si es necesario (para capas convolucionales)
        if len(features.shape) > 2:
            original_shape = features.shape
            features = features.reshape(features.shape[0], -1)
            logger.info(f"📐 Características aplanadas: {original_shape} -> {features.shape}")
        
        logger.info(f"✅ Características extraídas: {features.shape}")
        
        # 3. Calcular media
        logger.info("📊 Calculando media de características...")
        self.mean = np.mean(features, axis=0)
        
        # 4. Calcular covarianza usando Empirical Covariance (robusto)
        logger.info("📊 Calculando matriz de covarianza...")
        cov_estimator = EmpiricalCovariance()
        cov_estimator.fit(features)
        self.cov = cov_estimator.covariance_
        
        # 5. Calcular matriz de covarianza inversa (pseudo-inversa para robustez)
        logger.info("🔢 Calculando matriz de covarianza inversa...")
        self.inv_cov = np.linalg.pinv(self.cov)
        
        # 6. Calcular threshold basado en las distancias de entrenamiento
        logger.info(f"📏 Calculando threshold (percentil {percentile})...")
        distances = []
        
        # Calcular distancias en batches para eficiencia
        batch_size = 100
        for i in range(0, len(features), batch_size):
            batch = features[i:i+batch_size]
            for feature in batch:
                dist = mahalanobis(feature, self.mean, self.inv_cov)
                distances.append(dist)
        
        distances = np.array(distances)
        self.threshold = np.percentile(distances, percentile)
        
        # 7. Estadísticas finales
        self.is_fitted = True
        
        stats = {
            'n_samples': len(train_images),
            'feature_dim': features.shape[1],
            'threshold': float(self.threshold),
            'percentile': percentile,
            'mean_distance': float(np.mean(distances)),
            'std_distance': float(np.std(distances)),
            'min_distance': float(np.min(distances)),
            'max_distance': float(np.max(distances))
        }
        
        logger.info("✅ OOD Detector entrenado exitosamente")
        logger.info(f"📊 Threshold calculado: {self.threshold:.2f}")
        logger.info(f"📊 Distancia promedio: {stats['mean_distance']:.2f} ± {stats['std_distance']:.2f}")
        
        return stats
        
    def predict(self, image):
        """
        Predice si una imagen está dentro o fuera de distribución
        
        Args:
            image: Imagen preprocesada (224, 224, 3) con valores [0, 1]
                  o batch de imágenes (n, 224, 224, 3)
            
        Returns:
            dict: {
                'is_valid': bool,       # True si es imagen válida de lesión cutánea
                'distance': float,      # Distancia de Mahalanobis
                'threshold': float,     # Threshold usado
                'confidence': float,    # Confianza de que es válida (0-1)
                'message': str,         # Mensaje para el usuario
                'severity': str         # 'valid', 'warning', 'rejected'
            }
        """
        if not self.is_fitted:
            logger.warning("⚠️ OOD Detector no entrenado. Aceptando todas las imágenes.")
            return {
                'is_valid': True,
                'distance': 0.0,
                'threshold': 0.0,
                'confidence': 1.0,
                'message': '✅ Validación OOD deshabilitada (detector no entrenado)',
                'severity': 'valid'
            }
        
        # Expandir dimensión de batch si es necesario
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Extraer características
        features = self.feature_extractor.predict(image, verbose=0)
        
        # Aplanar si es necesario
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        # Calcular distancia de Mahalanobis
        distance = mahalanobis(features[0], self.mean, self.inv_cov)
        
        # Decidir si es válida
        is_valid = distance <= self.threshold
        
        # Calcular confianza (normalizada e invertida)
        # Confianza alta = distancia baja
        # Usamos una función sigmoide invertida para suavizar
        normalized_distance = distance / self.threshold
        confidence = max(0.0, min(1.0, 1.0 / (1.0 + normalized_distance)))
        
        # Determinar severidad y mensaje
        if distance <= self.threshold:
            severity = 'valid'
            message = "✅ Imagen válida: parece una lesión cutánea"
        elif distance <= self.threshold * 1.5:
            severity = 'warning'
            message = "⚠️ Imagen sospechosa: La calidad de la imagen podría no ser óptima. Intenta con una foto más clara de la lesión cutánea."
        else:
            severity = 'rejected'
            message = "❌ Imagen rechazada: Esta imagen NO parece una lesión cutánea. Por favor, sube una foto de una lesión de piel."
        
        result = {
            'is_valid': is_valid,
            'distance': float(distance),
            'threshold': float(self.threshold),
            'confidence': float(confidence),
            'message': message,
            'severity': severity,
            'ratio': float(distance / self.threshold)  # Para debugging
        }
        
        logger.info(f"🔍 OOD Check: distance={distance:.2f}, threshold={self.threshold:.2f}, "
                   f"ratio={result['ratio']:.2f}, valid={is_valid}")
        
        return result
    
    def save(self, filepath):
        """
        Guarda las estadísticas del detector en un archivo .npz
        
        Args:
            filepath: Ruta donde guardar el archivo (e.g., 'models/ood_stats.npz')
        """
        if not self.is_fitted:
            raise ValueError("❌ No se puede guardar un detector no entrenado. Ejecuta fit() primero.")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        np.savez(
            filepath,
            mean=self.mean,
            cov=self.cov,
            inv_cov=self.inv_cov,
            threshold=self.threshold,
            layer_name=self.layer_name
        )
        
        logger.info(f"💾 OOD Detector guardado en: {filepath}")
    
    def load(self, filepath):
        """
        Carga las estadísticas del detector desde un archivo .npz
        
        Args:
            filepath: Ruta del archivo a cargar (e.g., 'models/ood_stats.npz')
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"❌ Archivo no encontrado: {filepath}")
        
        data = np.load(filepath, allow_pickle=True)
        self.mean = data['mean']
        self.cov = data['cov']
        self.inv_cov = data['inv_cov']
        self.threshold = float(data['threshold'])
        
        # Verificar layer_name si existe en el archivo
        if 'layer_name' in data:
            saved_layer_name = str(data['layer_name'])
            if saved_layer_name != self.layer_name:
                logger.warning(f"⚠️ Layer name mismatch: guardado='{saved_layer_name}', actual='{self.layer_name}'")
        
        self.is_fitted = True
        
        logger.info(f"📂 OOD Detector cargado desde: {filepath}")
        logger.info(f"📊 Threshold: {self.threshold:.2f}")


# Función de utilidad para uso rápido
def quick_ood_check(model, image, stats_path='models/ood_detector_stats.npz'):
    """
    Función de utilidad para hacer una verificación rápida de OOD
    
    Args:
        model: Modelo de Keras cargado
        image: Imagen preprocesada (224, 224, 3)
        stats_path: Ruta al archivo de estadísticas del OOD detector
    
    Returns:
        dict: Resultado de la predicción OOD
    """
    if not os.path.exists(stats_path):
        logger.warning(f"⚠️ OOD stats no encontradas en: {stats_path}")
        return {
            'is_valid': True,
            'distance': 0.0,
            'threshold': 0.0,
            'confidence': 1.0,
            'message': '✅ Validación OOD no disponible',
            'severity': 'valid'
        }
    
    detector = OODDetector(model)
    detector.load(stats_path)
    return detector.predict(image)
