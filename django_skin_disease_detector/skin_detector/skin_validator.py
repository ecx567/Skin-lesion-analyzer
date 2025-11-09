"""
Sistema de Validación de Imágenes de Piel (Skin Validator)

Sistema híbrido que combina múltiples técnicas para detectar
imágenes que NO son de lesiones cutáneas:

1. Análisis de Color: Detecta tonos de piel humana
2. Análisis de Textura: Detecta patrones de piel vs objetos
3. Confianza del Modelo: Rechaza predicciones muy inciertas
4. Entropía de Predicción: Detecta distribuciones anómalas

Este enfoque es más robusto que solo usar Mahalanobis Distance.
"""

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import logging
from typing import Dict, Tuple
import os

logger = logging.getLogger(__name__)


class SkinValidator:
    """
    Validador híbrido de imágenes cutáneas
    
    Combina múltiples técnicas para determinar si una imagen
    contiene una lesión cutánea real o es un objeto/animal/paisaje.
    """
    
    def __init__(self, model=None):
        """
        Inicializa el validador
        
        Args:
            model: Modelo de clasificación (opcional, para análisis de confianza)
        """
        self.model = model
        self.is_enabled = True
        
        # Rangos de color de piel HUMANA en HSV
        # MÁS ESTRICTOS para evitar falsos positivos con animales
        self.skin_hsv_ranges = [
            # Piel clara (tonos rosados/beige)
            {'lower': np.array([0, 15, 80], dtype=np.uint8),
             'upper': np.array([17, 170, 255], dtype=np.uint8)},
            # Piel media (tonos cálidos)
            {'lower': np.array([0, 25, 70], dtype=np.uint8),
             'upper': np.array([20, 180, 255], dtype=np.uint8)},
        ]
        
        # Rangos de color de piel en YCrCb (más preciso para piel humana)
        self.skin_ycrcb_ranges = [
            {'lower': np.array([0, 135, 85], dtype=np.uint8),
             'upper': np.array([255, 180, 135], dtype=np.uint8)},
        ]
        
        # Rangos de colores a RECHAZAR (pelo de animales, objetos)
        self.animal_hsv_ranges = [
            # Marrones/dorados (pelo de perro)
            {'lower': np.array([10, 100, 20], dtype=np.uint8),
             'upper': np.array([30, 255, 200], dtype=np.uint8)},
            # Grises (pelo de gato, cemento)
            {'lower': np.array([0, 0, 50], dtype=np.uint8),
             'upper': np.array([180, 50, 200], dtype=np.uint8)},
        ]
        
        # Umbrales de validación (ajustados para rechazar animales)
        self.min_skin_percentage = 5.0  # Mínimo de piel humana detectada
        self.max_animal_percentage = 30.0  # Máximo de colores de animal permitidos
        self.min_confidence = 0.15  # Confianza mínima del modelo
        self.max_entropy = 3.0  # Entropía máxima
        self.min_texture_variance = 50  # Varianza de textura mínima
        
        logger.info("✅ SkinValidator inicializado")
    
    def _analyze_skin_color(self, image: np.ndarray) -> Dict:
        """
        Analiza si la imagen contiene tonos de PIEL HUMANA
        y detecta colores de animales/objetos
        
        Args:
            image: Imagen en formato RGB, valores [0, 255], shape (H, W, 3)
        
        Returns:
            Dict con resultados del análisis de color
        """
        try:
            # Asegurar formato correcto
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
            
            # Convertir a HSV y YCrCb
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            
            # ==================================================
            # DETECCIÓN DE PIEL HUMANA (debe estar presente)
            # ==================================================
            skin_mask_hsv = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for range_dict in self.skin_hsv_ranges:
                mask = cv2.inRange(hsv, range_dict['lower'], range_dict['upper'])
                skin_mask_hsv = cv2.bitwise_or(skin_mask_hsv, mask)
            
            skin_mask_ycrcb = np.zeros(ycrcb.shape[:2], dtype=np.uint8)
            for range_dict in self.skin_ycrcb_ranges:
                mask = cv2.inRange(ycrcb, range_dict['lower'], range_dict['upper'])
                skin_mask_ycrcb = cv2.bitwise_or(skin_mask_ycrcb, mask)
            
            # Combinar máscaras (AND para ser más estricto)
            skin_mask = cv2.bitwise_and(skin_mask_hsv, skin_mask_ycrcb)
            
            # Limpiar ruido
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            
            # ==================================================
            # DETECCIÓN DE COLORES DE ANIMALES/OBJETOS (debe estar ausente)
            # ==================================================
            animal_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for range_dict in self.animal_hsv_ranges:
                mask = cv2.inRange(hsv, range_dict['lower'], range_dict['upper'])
                animal_mask = cv2.bitwise_or(animal_mask, mask)
            
            # Limpiar ruido
            animal_mask = cv2.morphologyEx(animal_mask, cv2.MORPH_CLOSE, kernel)
            
            # Calcular porcentajes
            total_pixels = skin_mask.shape[0] * skin_mask.shape[1]
            skin_pixels = np.sum(skin_mask > 0)
            animal_pixels = np.sum(animal_mask > 0)
            
            skin_percentage = (skin_pixels / total_pixels) * 100
            animal_percentage = (animal_pixels / total_pixels) * 100
            
            # Determinar si tiene características de piel humana
            has_skin = skin_percentage >= self.min_skin_percentage
            has_animal_colors = animal_percentage > self.max_animal_percentage
            
            return {
                'skin_percentage': skin_percentage,
                'animal_percentage': animal_percentage,
                'has_skin': has_skin,
                'has_animal_colors': has_animal_colors,
                'skin_pixels': int(skin_pixels),
                'animal_pixels': int(animal_pixels),
                'total_pixels': int(total_pixels)
            }
            
        except Exception as e:
            logger.error(f"Error en análisis de color: {str(e)}")
            # En caso de error, asumir que puede ser válida
            return {
                'skin_percentage': 100.0,
                'animal_percentage': 0.0,
                'has_skin': True,
                'has_animal_colors': False,
                'skin_pixels': 0,
                'animal_pixels': 0,
                'total_pixels': 0,
                'error': str(e)
            }
    
    def _analyze_texture(self, image: np.ndarray) -> Dict:
        """
        Analiza la textura de la imagen para detectar patrones de piel
        
        Args:
            image: Imagen en formato RGB, valores [0, 255]
        
        Returns:
            Dict con resultados del análisis de textura
        """
        try:
            # Convertir a escala de grises
            if image.max() <= 1.0:
                gray = (image[:, :, 0] * 255).astype(np.uint8)
            else:
                gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Calcular varianza de textura (indicador de detalle)
            texture_variance = np.var(gray)
            
            # Calcular gradientes (bordes)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            edge_density = np.mean(gradient_magnitude)
            
            # Calcular estadísticas de intensidad
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # La piel tiene características específicas:
            # - Varianza moderada (no muy lisa, no muy texturizada)
            # - Bordes suaves (no muchos bordes fuertes)
            # - Distribución de intensidad específica
            
            has_skin_texture = (
                texture_variance > self.min_texture_variance and
                edge_density < 100 and  # No muchos bordes fuertes
                30 < mean_intensity < 230  # No completamente oscura o clara
            )
            
            return {
                'texture_variance': float(texture_variance),
                'edge_density': float(edge_density),
                'mean_intensity': float(mean_intensity),
                'std_intensity': float(std_intensity),
                'has_skin_texture': has_skin_texture
            }
            
        except Exception as e:
            logger.error(f"Error en análisis de textura: {str(e)}")
            return {
                'texture_variance': 1000.0,
                'edge_density': 0.0,
                'mean_intensity': 128.0,
                'std_intensity': 50.0,
                'has_skin_texture': True,
                'error': str(e)
            }
    
    def _analyze_prediction_confidence(self, predictions: np.ndarray) -> Dict:
        """
        Analiza la confianza y entropía de las predicciones del modelo
        
        Args:
            predictions: Array de probabilidades [batch_size, num_classes]
        
        Returns:
            Dict con análisis de confianza
        """
        try:
            # Obtener predicción más probable
            max_confidence = float(np.max(predictions))
            
            # Calcular entropía de Shannon (mide incertidumbre)
            # Entropía alta = modelo muy inseguro = posible OOD
            epsilon = 1e-10  # Para evitar log(0)
            entropy = -np.sum(predictions * np.log(predictions + epsilon))
            
            # Calcular diferencia entre top-1 y top-2
            sorted_preds = np.sort(predictions)[::-1]
            confidence_gap = float(sorted_preds[0] - sorted_preds[1])
            
            is_confident = (
                max_confidence >= self.min_confidence and
                entropy <= self.max_entropy
            )
            
            return {
                'max_confidence': max_confidence,
                'entropy': float(entropy),
                'confidence_gap': confidence_gap,
                'is_confident': is_confident
            }
            
        except Exception as e:
            logger.error(f"Error en análisis de confianza: {str(e)}")
            return {
                'max_confidence': 1.0,
                'entropy': 0.0,
                'confidence_gap': 1.0,
                'is_confident': True,
                'error': str(e)
            }
    
    def validate(self, image: np.ndarray, predictions: np.ndarray = None) -> Dict:
        """
        Valida si una imagen es de una lesión cutánea
        
        Combina análisis de color, textura y confianza del modelo
        para determinar si la imagen es válida.
        
        Args:
            image: Imagen preprocesada, shape (224, 224, 3), valores [0, 1]
            predictions: Predicciones del modelo (opcional), shape (num_classes,)
        
        Returns:
            Dict con resultado de validación:
            {
                'is_valid': bool,
                'confidence_score': float (0-100),
                'message': str,
                'details': dict con análisis detallado
            }
        """
        if not self.is_enabled:
            return {
                'is_valid': True,
                'confidence_score': 100.0,
                'message': 'Validación deshabilitada',
                'details': {}
            }
        
        try:
            # Convertir imagen a formato correcto para análisis
            if image.max() <= 1.0:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                image_uint8 = image.astype(np.uint8)
            
            # 1. Análisis de color de piel
            color_analysis = self._analyze_skin_color(image_uint8)
            
            # 2. Análisis de textura
            texture_analysis = self._analyze_texture(image_uint8)
            
            # 3. Análisis de confianza (si hay predicciones)
            confidence_analysis = None
            if predictions is not None:
                confidence_analysis = self._analyze_prediction_confidence(predictions)
            
            # Calcular score de confianza (0-100)
            scores = []
            weights = []
            
            # Score de color (35% de peso, reducido)
            if color_analysis['has_skin']:
                color_score = min(100, (color_analysis['skin_percentage'] / 30) * 100)
            else:
                # Penalización moderada si no detecta piel
                color_score = max(30, color_analysis['skin_percentage'] * 3)
            scores.append(color_score)
            weights.append(0.35)
            
            # Score de textura (25% de peso, reducido)
            if texture_analysis['has_skin_texture']:
                texture_score = 100
            else:
                # No penalizar tanto la textura
                texture_score = 70
            scores.append(texture_score)
            weights.append(0.25)
            
            # Score de confianza (40% de peso, AUMENTADO)
            # Este es el más importante para distinguir animales
            if confidence_analysis and confidence_analysis['is_confident']:
                conf_score = min(100, confidence_analysis['max_confidence'] * 120)
            elif confidence_analysis:
                # Si el modelo tiene baja confianza, usar score basado en confianza
                conf_score = max(40, confidence_analysis['max_confidence'] * 200)
            else:
                conf_score = 100  # Si no hay predicciones, no penalizar
            scores.append(conf_score)
            weights.append(0.40)
            
            # Calcular score ponderado
            confidence_score = np.average(scores, weights=weights)
            
            # ===============================================
            # DECISIÓN DE VALIDACIÓN (NUEVA LÓGICA ESTRICTA)
            # ===============================================
            
            # REGLA 1: Si tiene colores de animal, RECHAZAR inmediatamente
            if color_analysis['has_animal_colors']:
                is_valid = False
                confidence_score = 20
                reason = "colores de animal detectados"
            
            # REGLA 2: Si NO tiene piel humana Y confianza baja, RECHAZAR
            elif not color_analysis['has_skin'] and confidence_analysis:
                if confidence_analysis['max_confidence'] < 0.25:
                    is_valid = False
                    confidence_score = min(confidence_score, 30)
                    reason = "sin tonos de piel humana"
                else:
                    is_valid = True
                    reason = "confianza aceptable"
            
            # REGLA 3: Score muy bajo, RECHAZAR
            elif confidence_score < 35:
                is_valid = False
                reason = "score de validación bajo"
            
            # REGLA 4: Todo OK, ACEPTAR
            else:
                is_valid = True
                reason = "validación exitosa"
            
            # Mensaje descriptivo
            if is_valid:
                message = "Imagen válida: Parece una lesión cutánea"
            else:
                # Mensajes específicos según la razón
                if color_analysis['has_animal_colors']:
                    message = f"Imagen rechazada: Detectados colores de animal/objeto ({color_analysis['animal_percentage']:.1f}% de la imagen)"
                elif not color_analysis['has_skin']:
                    message = f"Imagen rechazada: Sin tonos de piel humana ({color_analysis['skin_percentage']:.1f}% detectado)"
                elif confidence_analysis and confidence_analysis['max_confidence'] < 0.15:
                    message = f"Imagen rechazada: Confianza muy baja ({confidence_analysis['max_confidence']*100:.1f}%)"
                else:
                    message = f"Imagen rechazada: No parece una lesión cutánea (score: {confidence_score:.1f})"
            
            return {
                'is_valid': is_valid,
                'confidence_score': float(confidence_score),
                'message': message,
                'details': {
                    'color_analysis': color_analysis,
                    'texture_analysis': texture_analysis,
                    'confidence_analysis': confidence_analysis,
                    'scores': {
                        'color': float(color_score),
                        'texture': float(texture_score),
                        'confidence': float(conf_score)
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error en validación: {str(e)}")
            # En caso de error, permitir la imagen para no bloquear el sistema
            return {
                'is_valid': True,
                'confidence_score': 50.0,
                'message': f'Error en validación: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def set_thresholds(self, 
                      min_skin_percentage: float = None,
                      min_confidence: float = None,
                      max_entropy: float = None,
                      min_texture_variance: float = None):
        """
        Ajusta los umbrales de validación
        
        Args:
            min_skin_percentage: Porcentaje mínimo de piel (0-100)
            min_confidence: Confianza mínima del modelo (0-1)
            max_entropy: Entropía máxima permitida
            min_texture_variance: Varianza mínima de textura
        """
        if min_skin_percentage is not None:
            self.min_skin_percentage = min_skin_percentage
            logger.info(f"Umbral de piel actualizado: {min_skin_percentage}%")
        
        if min_confidence is not None:
            self.min_confidence = min_confidence
            logger.info(f"Confianza mínima actualizada: {min_confidence}")
        
        if max_entropy is not None:
            self.max_entropy = max_entropy
            logger.info(f"Entropía máxima actualizada: {max_entropy}")
        
        if min_texture_variance is not None:
            self.min_texture_variance = min_texture_variance
            logger.info(f"Varianza de textura actualizada: {min_texture_variance}")
    
    def disable(self):
        """Deshabilita temporalmente la validación"""
        self.is_enabled = False
        logger.warning("⚠️ SkinValidator DESHABILITADO")
    
    def enable(self):
        """Habilita la validación"""
        self.is_enabled = True
        logger.info("✅ SkinValidator HABILITADO")
