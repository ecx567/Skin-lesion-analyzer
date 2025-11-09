# 🔍 OOD Detection - Sistema de Validación de Imágenes

## 📋 Descripción

El **OOD (Out-of-Distribution) Detection** es un sistema de validación que **rechaza automáticamente imágenes que NO son de lesiones cutáneas**. Esto evita que usuarios suban accidentalmente imágenes incorrectas (paisajes, animales, objetos, etc.) que podrían dar resultados sin sentido.

## 🎯 ¿Qué hace?

El sistema **analiza la imagen ANTES de hacer la predicción** y determina si:
- ✅ **ES** una imagen de lesión cutánea → Continúa con la predicción
- ❌ **NO ES** una imagen de lesión cutánea → Rechaza la imagen con un mensaje claro

## 📊 Método Utilizado

**Mahalanobis Distance OOD Detection**

Basado en el paper: "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks" (NeurIPS 2018)
- Paper: https://arxiv.org/abs/1807.03888
- Repositorio: https://github.com/pokaxxxxxxxxxxx/deep_Mahalanobis_detector

### ¿Cómo funciona?

1. **Entrenamiento**: Calcula estadísticas (media y covarianza) de las características del modelo usando el dataset HAM10000
2. **Validación**: Para cada imagen nueva, calcula qué tan "diferente" es de las imágenes de entrenamiento
3. **Decisión**: Si la imagen es muy diferente → se rechaza

## 🚀 Instalación y Configuración

### Paso 1: Verificar dependencias

Las dependencias ya están en `requirements.txt`:
```bash
scipy>=1.11.0  # Para mahalanobis distance
scikit-learn>=1.7.0
```

Si no las tienes instaladas:
```bash
pip install scipy scikit-learn
```

### Paso 2: Entrenar el OOD Detector

El OOD Detector necesita ser entrenado **UNA VEZ** con imágenes del dataset HAM10000:

```bash
# Desde la carpeta django_skin_disease_detector/
python train_ood_detector.py
```

**¿Qué hace este comando?**
1. Carga el modelo de clasificación (`improved_balanced_7class_model.h5`)
2. Carga 1,500 imágenes del dataset HAM10000 (balanceadas)
3. Extrae características de la penúltima capa del modelo
4. Calcula estadísticas (media, covarianza, threshold)
5. Guarda el archivo: `models/ood_detector_stats.npz`

**Tiempo estimado:** 2-5 minutos (depende de tu hardware)

**Requisitos:**
- Dataset HAM10000 en: `../ai-model/datasets/ham10000/`
- Modelo entrenado en: `models/improved_balanced_7class_model.h5`

### Paso 3: ¡Listo! El sistema ya está activo

Una vez entrenado, el OOD Detector se carga automáticamente con el modelo de predicción.

## 📝 Uso

### Para Usuarios Finales

**Simplemente sube una imagen como siempre:**

1. Si subes una **lesión cutánea** → ✅ Funciona normal
2. Si subes **cualquier otra cosa** → ❌ Verás un mensaje de error:

```
❌ Imagen rechazada: Esta imagen NO parece una lesión cutánea.
Por favor, sube una foto de una lesión de piel.

Recomendaciones:
📸 Asegúrate de fotografiar una lesión cutánea real
💡 Usa buena iluminación natural
🎯 Enfoca claramente la lesión
```

### Para Desarrolladores

**El OOD Detector se integra automáticamente en el flujo de predicción:**

```python
# En predictor.py
result = predictor.predict(image_path)

# Si la imagen es rechazada:
if result.get('success') == False and result.get('error') == 'invalid_image_ood':
    # La imagen NO es una lesión cutánea
    print(result['message'])  # Mensaje de error
    print(result['ood_validation'])  # Detalles técnicos
```

**Respuesta de error OOD:**
```python
{
    'success': False,
    'error': 'invalid_image_ood',
    'error_type': 'out_of_distribution',
    'message': '❌ Imagen rechazada: Esta imagen NO parece una lesión cutánea...',
    'ood_validation': {
        'is_valid': False,
        'distance': 45.23,  # Distancia de Mahalanobis
        'threshold': 25.80,  # Threshold calculado
        'confidence': 0.12,  # Confianza de que ES válida (0-1)
        'severity': 'rejected',  # 'valid', 'warning', o 'rejected'
        'ratio': 1.75  # distance/threshold (>1 = rechazada)
    },
    'processing_time': 0.234
}
```

## 🔧 Configuración Avanzada

### Deshabilitar OOD Detection (si es necesario)

Si por alguna razón necesitas deshabilitar el OOD Detection temporalmente:

**Opción 1: Renombrar el archivo de stats**
```bash
mv models/ood_detector_stats.npz models/ood_detector_stats.npz.bak
```

**Opción 2: Eliminar el archivo**
```bash
rm models/ood_detector_stats.npz
```

El sistema detectará automáticamente que no existe y continuará funcionando sin validación OOD.

### Re-entrenar con diferentes parámetros

Puedes editar `train_ood_detector.py` y modificar:

```python
SAMPLE_SIZE = 1500      # Número de imágenes (más = mejor pero más lento)
MAX_PER_CLASS = 250     # Balance de clases
PERCENTILE = 95         # Threshold (95 = acepta 95% del training)
LAYER_NAME = 'dense'    # Capa para extraer características
```

## 📊 Resultados Esperados

### Imágenes que ACEPTARÁ ✅
- Melanoma (MEL)
- Nevus melanocítico (NV)
- Carcinoma basocelular (BCC)
- Queratosis actínica (AKIEC)
- Queratosis benigna (BKL)
- Dermatofibroma (DF)
- Lesiones vasculares (VASC)
- **Cualquier lesión cutánea similar**

### Imágenes que RECHAZARÁ ❌
- Fotos de paisajes
- Fotos de animales
- Fotos de objetos
- Fotos de comida
- Fotos de otras partes del cuerpo (ojos, manos, etc.)
- Imágenes de muy baja calidad
- Imágenes sin contenido médico

## 🧪 Testing

### Probar manualmente

1. **Sube una imagen de lesión cutánea del dataset** → Debe aceptarse
2. **Sube una foto de un gato** → Debe rechazarse
3. **Sube una foto de un paisaje** → Debe rechazarse

### Test programático

```python
from skin_detector.predictor import get_predictor

predictor = get_predictor()

# Test 1: Imagen válida
result = predictor.predict('path/to/skin_lesion.jpg')
assert result['success'] == True

# Test 2: Imagen inválida
result = predictor.predict('path/to/cat.jpg')
assert result['success'] == False
assert result['error'] == 'invalid_image_ood'
```

## 📈 Métricas de Rendimiento

Después de entrenar, verás algo como:

```
✅ Threshold calculado: 25.80
✅ Distancia promedio: 18.45 ± 5.23
✅ Rango de distancias: [3.21, 42.67]
```

**Interpretación:**
- **Threshold**: Valor límite para aceptar/rechazar
- **Distancia promedio**: Qué tan "diferentes" son las imágenes entre sí
- **Rango**: Mínimo y máximo de distancias en el training set

## ⚠️ Consideraciones Importantes

### 1. **No Afecta Funcionalidad Existente**
- ✅ Si el OOD Detector no está entrenado → el sistema funciona normalmente
- ✅ Si falla la validación OOD → se registra un warning y continúa
- ✅ Código 100% **backward compatible**

### 2. **Falsos Positivos/Negativos**
- El sistema puede ocasionalmente:
  - ❌ Rechazar una lesión cutánea válida (raro)
  - ✅ Aceptar una imagen similar a piel (raro)
- Solución: Ajustar `PERCENTILE` en el entrenamiento

### 3. **Rendimiento**
- La validación OOD añade ~50-100ms al tiempo de predicción
- Es un overhead mínimo comparado con el beneficio

## 🔍 Troubleshooting

### "OOD Detector no encontrado"
```
ℹ️ OOD Detector no encontrado. Las imágenes no serán validadas.
   Para habilitar, ejecuta: python train_ood_detector.py
```
**Solución:** Ejecuta `python train_ood_detector.py` para entrenar el detector.

### "Error cargando OOD Detector"
**Posibles causas:**
1. Archivo `ood_detector_stats.npz` corrupto → Re-entrenar
2. Incompatibilidad de versión de NumPy → Reinstalar scipy
3. Modelo del clasificador cambió → Re-entrenar OOD Detector

**Solución rápida:** Elimina `models/ood_detector_stats.npz` y re-entrena.

### "Rechaza imágenes válidas"
**Causas:**
- Threshold muy bajo (percentil muy bajo)
- Pocas imágenes de entrenamiento

**Solución:**
1. Aumenta `PERCENTILE` a 97 o 98 en `train_ood_detector.py`
2. Aumenta `SAMPLE_SIZE` a 2000+ imágenes
3. Re-entrena el detector

### "Acepta imágenes inválidas"
**Causas:**
- Threshold muy alto
- Imágenes de entrenamiento con mucha variabilidad

**Solución:**
1. Reduce `PERCENTILE` a 90 o 92
2. Re-entrena el detector

## 📚 Referencias

1. **Paper Original:**
   - Kimin Lee et al. "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks" (NeurIPS 2018)
   - https://arxiv.org/abs/1807.03888

2. **Implementación de Referencia:**
   - https://github.com/pokaxxxxxxxxxxx/deep_Mahalanobis_detector

3. **Artículos Relacionados:**
   - ODIN: https://arxiv.org/abs/1706.02690
   - Outlier Exposure: https://arxiv.org/abs/1812.04606

## 🎯 Conclusión

El OOD Detection es una **capa de seguridad adicional** que mejora significativamente la experiencia del usuario al:
- ✅ Prevenir diagnósticos erróneos por imágenes incorrectas
- ✅ Guiar al usuario con mensajes claros
- ✅ Mantener la integridad del sistema
- ✅ No afectar la funcionalidad existente

¡Es completamente opcional pero altamente recomendado para producción! 🚀
