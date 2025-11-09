import os
import time
import numpy as np
from PIL import Image
import tensorflow as tf
from django.conf import settings
from typing import Dict, Tuple, Optional
import logging

# Configurar logging
logger = logging.getLogger(__name__)

# Importar Skin Validator (nuevo sistema de validación)
try:
    from .skin_validator import SkinValidator
    VALIDATOR_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ Skin Validator no disponible. Las imágenes no serán validadas.")
    VALIDATOR_AVAILABLE = False


class SkinDiseasePredictor:
    """
    Clase para manejar la predicción de enfermedades cutáneas usando el modelo H5
    """
    
    def __init__(self):
        self.model = None
        self.skin_validator = None  # Nuevo validador de piel
        self.validator_enabled = False  # Flag para validación
        self.img_size = 224  # Tamaño usado en el entrenamiento
        
        # Definición de las 7 clases (mismo orden que en el entrenamiento)
        self.class_names = {
            0: {'code': 'akiec', 'name': 'Actinic keratoses', 'spanish': 'Queratosis actínicas'},
            1: {'code': 'bcc', 'name': 'Basal cell carcinoma', 'spanish': 'Carcinoma basocelular'},
            2: {'code': 'bkl', 'name': 'Benign keratosis', 'spanish': 'Queratosis benigna'},
            3: {'code': 'df', 'name': 'Dermatofibroma', 'spanish': 'Dermatofibroma'},
            4: {'code': 'mel', 'name': 'Melanoma', 'spanish': 'Melanoma'},
            5: {'code': 'nv', 'name': 'Melanocytic nevi', 'spanish': 'Nevos melanocíticos'},
            6: {'code': 'vasc', 'name': 'Vascular lesions', 'spanish': 'Lesiones vasculares'}
        }
        
        # Información médica adicional sobre cada enfermedad
        self.disease_info = {
            'akiec': {
                'severity': 'Moderada',
                'description': 'Lesiones precancerosas causadas por daño solar crónico.',
                'recommendation': 'Consulte a un dermatólogo para evaluación y tratamiento.',
                'risk_level': 3
            },
            'bcc': {
                'severity': 'Alta',
                'description': 'Tipo más común de cáncer de piel, generalmente de crecimiento lento.',
                'recommendation': 'Requiere atención médica inmediata. Consulte a un oncólogo dermatólogo.',
                'risk_level': 4
            },
            'bkl': {
                'severity': 'Baja',
                'description': 'Lesión benigna común, no cancerosa.',
                'recommendation': 'Generalmente no requiere tratamiento, pero monitoree cambios.',
                'risk_level': 1
            },
            'df': {
                'severity': 'Baja',
                'description': 'Tumor benigno del tejido conectivo de la piel.',
                'recommendation': 'Lesión benigna. Consulte si hay cambios o molestias.',
                'risk_level': 1
            },
            'mel': {
                'severity': 'Muy Alta',
                'description': 'Forma más agresiva de cáncer de piel.',
                'recommendation': '⚠️ URGENTE: Consulte inmediatamente a un oncólogo dermatólogo.',
                'risk_level': 5
            },
            'nv': {
                'severity': 'Baja',
                'description': 'Lunares comunes, generalmente benignos.',
                'recommendation': 'Monitoree cambios. Consulte si nota alteraciones en forma, color o tamaño.',
                'risk_level': 1
            },
            'vasc': {
                'severity': 'Baja a Moderada',
                'description': 'Lesiones relacionadas con vasos sanguíneos de la piel.',
                'recommendation': 'Consulte a un dermatólogo para evaluación apropiada.',
                'risk_level': 2
            }
        }
        
        # Cargar modelo al inicializar
        self._load_model()
    
    def _load_model(self):
        """Cargar el modelo H5 entrenado con Keras 3.x y TensorFlow 2.20+"""
        try:
            model_path = os.path.join(settings.BASE_DIR, 'models', 'improved_balanced_7class_model.h5')
            
            logger.info(f"Intentando cargar modelo desde: {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
            
            # Verificar tamaño del archivo
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            logger.info(f"Tamaño del archivo del modelo: {file_size_mb:.2f} MB")
            
            # Cargar modelo con Keras 3.x
            # El modelo fue entrenado con Focal Loss personalizada, así que cargamos sin compilar
            logger.info("Cargando modelo con Keras 3.x (sin compilar)...")
            self.model = tf.keras.models.load_model(model_path, compile=False)
            
            # Recompilar con pérdida estándar para predicción
            logger.info("Recompilando modelo con pérdida estándar...")
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Verificar la arquitectura del modelo
            logger.info(f"✅ Modelo cargado exitosamente!")
            logger.info(f"   - Capas totales: {len(self.model.layers)}")
            logger.info(f"   - Input shape: {self.model.input_shape}")
            logger.info(f"   - Output shape: {self.model.output_shape}")
            logger.info(f"   - Parámetros totales: {self.model.count_params():,}")
            
            # Hacer una predicción de prueba
            test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
            test_output = self.model.predict(test_input, verbose=0)
            logger.info(f"   - Test prediction shape: {test_output.shape}")
            logger.info(f"   - Test prediction sum: {np.sum(test_output):.4f}")
            
            if test_output.shape[1] != 7:
                raise ValueError(f"El modelo no tiene 7 salidas. Tiene: {test_output.shape[1]}")
            
            logger.info("🎉 Modelo listo para usar!")
            
            # Inicializar nuevo Skin Validator
            self._init_skin_validator()
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {str(e)}")
            logger.error(f"   Tipo de error: {type(e).__name__}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            raise Exception(f"No se pudo cargar el modelo: {str(e)}")
    
    def _init_skin_validator(self):
        """Inicializar el nuevo Skin Validator (sistema híbrido)"""
        if not VALIDATOR_AVAILABLE:
            logger.info("ℹ️ Skin Validator no disponible (módulo no importado)")
            return
        
        try:
            logger.info("🔍 Inicializando Skin Validator...")
            self.skin_validator = SkinValidator(model=self.model)
            self.validator_enabled = True
            
            logger.info("✅ Skin Validator activado")
            logger.info("   - Análisis de color de piel: ✓")
            logger.info("   - Análisis de textura: ✓")
            logger.info("   - Análisis de confianza: ✓")
            
        except Exception as e:
            logger.warning(f"⚠️ No se pudo inicializar Skin Validator: {str(e)}")
            logger.warning("   El sistema funcionará normalmente sin validación")
            self.skin_validator = None
            self.validator_enabled = False
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocesar imagen para el modelo
        """
        try:
            # Cargar imagen
            image = Image.open(image_path)
            
            # Convertir a RGB si es necesario
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Redimensionar a tamaño del modelo
            image = image.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
            
            # Convertir a array numpy
            image_array = np.array(image, dtype=np.float32)
            
            # Normalizar píxeles (0-1)
            image_array = image_array / 255.0
            
            # Agregar dimensión de batch
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocesando imagen: {str(e)}")
            raise Exception(f"Error al procesar la imagen: {str(e)}")
    
    def predict(self, image_path: str) -> Dict:
        """
        Realizar predicción en una imagen con manejo robusto de errores
        y validación OOD opcional
        """
        if self.model is None:
            # Intentar recargar modelo si es None
            logger.warning("Modelo no disponible, intentando recargar...")
            try:
                self._load_model()
            except Exception as e:
                logger.error(f"No se pudo recargar el modelo: {str(e)}")
                return self._get_dummy_prediction()
        
        start_time = time.time()
        
        try:
            # Preprocesar imagen
            processed_image = self.preprocess_image(image_path)
            
            # ========================================
            # PREDICCIÓN DEL MODELO
            # ========================================
            
            # Verificar que el modelo está disponible
            if self.model is None:
                return self._get_dummy_prediction()
            
            # Realizar predicción con manejo de errores
            try:
                predictions = self.model.predict(processed_image, verbose=0)
            except Exception as pred_error:
                logger.error(f"Error en model.predict: {str(pred_error)}")
                return self._get_dummy_prediction()
            
            # Verificar que las predicciones tienen el formato correcto
            if predictions is None or len(predictions) == 0:
                logger.error("Predicciones vacías del modelo")
                return self._get_dummy_prediction()
            
            # ========================================
            # VALIDACIÓN CON SKIN VALIDATOR (NUEVO)
            # ========================================
            validation_result = None
            if self.validator_enabled and self.skin_validator is not None:
                logger.info("🔍 Validando imagen con Skin Validator...")
                
                try:
                    # Validar imagen con análisis de color, textura y confianza
                    validation_result = self.skin_validator.validate(
                        image=processed_image[0],
                        predictions=predictions[0]
                    )
                    
                    # Si la imagen NO es válida, retornar error
                    if not validation_result['is_valid']:
                        logger.warning(f"❌ Imagen rechazada por Skin Validator")
                        logger.warning(f"   Razón: {validation_result['message']}")
                        logger.warning(f"   Score: {validation_result['confidence_score']:.1f}/100")
                        
                        details = validation_result.get('details', {})
                        if 'color_analysis' in details:
                            logger.warning(f"   Color: {details['color_analysis']['skin_percentage']:.1f}% piel")
                        if 'confidence_analysis' in details and details['confidence_analysis']:
                            logger.warning(f"   Confianza: {details['confidence_analysis']['max_confidence']*100:.1f}%")
                        
                        return {
                            'success': False,
                            'error': 'invalid_image_validator',
                            'error_type': 'not_skin_lesion',
                            'message': validation_result['message'],
                            'validation': {
                                'is_valid': False,
                                'confidence_score': validation_result['confidence_score'],
                                'details': details
                            },
                            'processing_time': round(time.time() - start_time, 3)
                        }
                    
                    logger.info(f"✅ Imagen válida (score: {validation_result['confidence_score']:.1f}/100)")
                    
                except Exception as val_error:
                    # Si falla validación, continuar (no romper el flujo)
                    logger.warning(f"⚠️ Error en validación: {str(val_error)}")
                    logger.warning("   Continuando sin validación...")
                    validation_result = None
            
            # Obtener probabilidades
            probabilities = predictions[0]
            
            # Verificar que tenemos 7 clases
            if len(probabilities) != 7:
                logger.error(f"Número incorrecto de clases predichas: {len(probabilities)}")
                return self._get_dummy_prediction()
            
            # Encontrar clase con mayor probabilidad
            predicted_class_idx = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class_idx])
            
            # Obtener información de la clase predicha
            predicted_class_info = self.class_names[predicted_class_idx]
            predicted_class_code = predicted_class_info['code']
            
            # Crear resultado detallado
            result = {
                'success': True,  # Agregado para consistencia
                'predicted_class': predicted_class_code,
                'confidence': confidence,
                'confidence_percentage': round(confidence * 100, 2),
                'class_name': predicted_class_info['name'],
                'class_name_spanish': predicted_class_info['spanish'],
                'processing_time': round(time.time() - start_time, 3),
                'all_probabilities': {},
                'disease_info': self.disease_info[predicted_class_code],
                'model_status': 'active'
            }
            
            # Agregar información de validación si existe
            if validation_result is not None:
                result['validation'] = {
                    'is_valid': validation_result['is_valid'],
                    'confidence_score': validation_result['confidence_score'],
                    'message': validation_result['message']
                }
            
            # Agregar todas las probabilidades
            for idx, prob in enumerate(probabilities):
                class_info = self.class_names[idx]
                result['all_probabilities'][class_info['code']] = {
                    'probability': float(prob),
                    'percentage': round(float(prob) * 100, 2),
                    'name': class_info['name'],
                    'spanish': class_info['spanish']
                }
            
            logger.info(f"Predicción exitosa: {predicted_class_code} ({confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            return self._get_dummy_prediction(error_msg=str(e))
    
    def _get_dummy_prediction(self, error_msg: str = None) -> Dict:
        """Crear predicción dummy cuando el modelo falla"""
        
        # Generar predicción aleatoria realista para demostración
        np.random.seed(42)  # Para resultados consistentes
        dummy_probs = np.random.dirichlet(np.ones(7), size=1)[0]
        predicted_class_idx = np.argmax(dummy_probs)
        confidence = float(dummy_probs[predicted_class_idx])
        
        predicted_class_info = self.class_names[predicted_class_idx]
        predicted_class_code = predicted_class_info['code']
        
        result = {
            'predicted_class': predicted_class_code,
            'confidence': confidence,
            'confidence_percentage': round(confidence * 100, 2),
            'class_name': predicted_class_info['name'],
            'class_name_spanish': predicted_class_info['spanish'],
            'processing_time': 0.1,
            'all_probabilities': {},
            'disease_info': self.disease_info[predicted_class_code],
            'model_status': 'dummy',
            'error_message': error_msg or 'Modelo no disponible - usando predicción de demostración'
        }
        
        # Agregar todas las probabilidades dummy
        for idx, prob in enumerate(dummy_probs):
            class_info = self.class_names[idx]
            result['all_probabilities'][class_info['code']] = {
                'probability': float(prob),
                'percentage': round(float(prob) * 100, 2),
                'name': class_info['name'],
                'spanish': class_info['spanish']
            }
        
        logger.warning(f"Usando predicción dummy: {predicted_class_code} ({confidence:.3f})")
        
        return result
    
    def get_top_predictions(self, image_path: str, top_n: int = 3) -> Dict:
        """
        Obtener las top N predicciones más probables
        """
        result = self.predict(image_path)
        
        # Ordenar probabilidades de mayor a menor
        sorted_probs = sorted(
            result['all_probabilities'].items(),
            key=lambda x: x[1]['probability'],
            reverse=True
        )
        
        # Tomar top N
        top_predictions = []
        for i, (class_code, prob_info) in enumerate(sorted_probs[:top_n]):
            top_predictions.append({
                'rank': i + 1,
                'class_code': class_code,
                'probability': prob_info['probability'],
                'percentage': prob_info['percentage'],
                'name': prob_info['name'],
                'spanish': prob_info['spanish'],
                'disease_info': self.disease_info[class_code]
            })
        
        result['top_predictions'] = top_predictions
        return result
    
    def validate_image(self, image_path: str) -> bool:
        """
        Validar si la imagen es adecuada para el análisis
        """
        try:
            image = Image.open(image_path)
            
            # Verificar dimensiones mínimas
            if image.size[0] < 50 or image.size[1] < 50:
                return False
                
            # Verificar que se puede convertir a RGB
            if image.mode not in ['RGB', 'RGBA', 'L']:
                return False
                
            return True
            
        except Exception:
            return False
    
    def get_model_summary(self) -> Dict:
        """
        Obtener resumen del modelo
        """
        if self.model is None:
            return {'error': 'Modelo no cargado'}
        
        try:
            return {
                'loaded': True,
                'total_layers': len(self.model.layers),
                'input_shape': str(self.model.input_shape),
                'output_shape': str(self.model.output_shape),
                'total_params': self.model.count_params(),
                'classes': list(self.class_names.keys()),
                'class_names': {k: v['spanish'] for k, v in self.class_names.items()}
            }
        except Exception as e:
            return {'error': str(e)}


# Instancia global del predictor (singleton)
_predictor_instance = None

def get_predictor() -> SkinDiseasePredictor:
    """
    Obtener instancia singleton del predictor
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = SkinDiseasePredictor()
    return _predictor_instance
