# 🧬 Entrenamiento del Modelo de IA - 7 Clases de Enfermedades de la Piel

## 📋 Índice
- [Descripción General](#descripción-general)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Clases de Enfermedades](#clases-de-enfermedades)
- [Estructura del Código](#estructura-del-código)
- [Funciones Principales](#funciones-principales)
- [Proceso de Entrenamiento](#proceso-de-entrenamiento)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Requisitos](#requisitos)
- [Uso](#uso)

---

## 🎯 Descripción General

**`improved_balanced_7class_training.py`** es el script principal para entrenar un modelo de Deep Learning que clasifica imágenes de lesiones cutáneas en 7 categorías diferentes de enfermedades de la piel.

### Características Principales:
- ✅ **Focal Loss**: Maneja el desbalanceo de clases en el dataset
- ✅ **Muestreo Equilibrado**: Asegura igual representación de todas las clases
- ✅ **Aumento de Datos Agresivo**: Genera variaciones de imágenes para mejorar generalización
- ✅ **Arquitectura Profunda**: CNN de 23 capas con BatchNormalization y Dropout
- ✅ **Generación de Datos Sintéticos**: Crea muestras adicionales para clases minoritarias
- ✅ **Conversión a TFLite**: Optimiza el modelo para dispositivos móviles

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE DE ENTRENAMIENTO                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  1. CARGA Y ANÁLISIS DE DATOS (HAM10000 Dataset)           │
│     - 10,015 imágenes dermoscópicas                         │
│     - Metadatos CSV con diagnósticos                        │
│     - 7 clases de enfermedades                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. BALANCEO DE DATOS                                       │
│     - Identificación de clases minoritarias                 │
│     - Oversampling (duplicación con variación)              │
│     - Objetivo: min 500 muestras por clase                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. AUMENTO DE DATOS (Data Augmentation)                    │
│     - Rotación: ±60°                                        │
│     - Desplazamiento: ±40%                                  │
│     - Zoom: ±50%                                            │
│     - Flips horizontal y vertical                           │
│     - Ajuste de brillo: 0.5-1.5x                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. CREACIÓN DEL MODELO CNN                                 │
│     - Input: 224x224x3 (RGB)                                │
│     - 5 bloques convolucionales                             │
│     - 23 capas totales                                      │
│     - Output: 7 clases (softmax)                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  5. COMPILACIÓN CON FOCAL LOSS                              │
│     - Optimizer: Adam (lr=0.001)                            │
│     - Loss: Focal Loss (α=0.25, γ=2.0)                     │
│     - Métricas: accuracy, precision, recall                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  6. ENTRENAMIENTO                                           │
│     - Épocas: 80                                            │
│     - Batch size: 28 (4 muestras × 7 clases)               │
│     - División: 70% train, 15% val, 15% test               │
│     - Callbacks: EarlyStopping, ModelCheckpoint, ReduceLR   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  7. EVALUACIÓN                                              │
│     - Métricas por clase                                    │
│     - Matriz de confusión                                   │
│     - Reporte de clasificación                              │
│     - Visualizaciones (gráficas de entrenamiento)           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  8. CONVERSIÓN A TFLITE                                     │
│     - Optimización: DEFAULT                                 │
│     - Compresión: ~3-5x                                     │
│     - Output: modelo para móviles                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🦠 Clases de Enfermedades

El modelo clasifica 7 tipos de lesiones cutáneas:

| Código | Nombre Completo | Descripción | Prevalencia |
|--------|----------------|-------------|-------------|
| **akiec** | Actinic Keratoses | Queratosis actínica - lesión precancerosa | Baja (~3%) |
| **bcc** | Basal Cell Carcinoma | Carcinoma basocelular - tipo de cáncer de piel | Baja (~5%) |
| **bkl** | Benign Keratosis | Queratosis benigna - lesión no cancerosa | Media (~11%) |
| **df** | Dermatofibroma | Tumor benigno del tejido conectivo | Muy baja (~1%) |
| **mel** | Melanoma | Melanoma - forma más peligrosa de cáncer de piel | Media (~11%) |
| **nv** | Melanocytic Nevi | Nevos melanocíticos - lunares comunes | Alta (~67%) |
| **vasc** | Vascular Lesions | Lesiones vasculares - malformaciones de vasos sanguíneos | Baja (~1%) |

### Problema de Desbalanceo:
- **nv** (nevos): ~6,705 imágenes (67%)
- **mel** (melanoma): ~1,113 imágenes (11%)
- **bkl**: ~1,099 imágenes (11%)
- **bcc**: ~514 imágenes (5%)
- **akiec**: ~327 imágenes (3%)
- **vasc**: ~142 imágenes (1%)
- **df**: ~115 imágenes (1%)

**Solución implementada**: Focal Loss + Oversampling

---

## 📦 Estructura del Código

### Clase Principal: `ImprovedBalanced7ClassModel`

```python
class ImprovedBalanced7ClassModel:
    def __init__(self, img_size=224):
        # Configuración inicial
        self.img_size = 224          # Tamaño de imagen de entrada
        self.num_classes = 7         # 7 enfermedades
        self.model = None            # Modelo CNN
        self.history = None          # Historial de entrenamiento
        self.class_names = {...}     # Diccionario de nombres de clases
        self.class_list = [...]      # Lista ordenada de códigos
```

---

## 🔧 Funciones Principales

### 1. `__init__(self, img_size=224)`
**Propósito**: Inicializa la clase con configuraciones básicas.

**Parámetros**:
- `img_size`: Tamaño de las imágenes (224x224 píxeles por defecto)

**Qué hace**:
- Define las 7 clases de enfermedades con sus nombres completos
- Inicializa variables para almacenar el modelo y el historial
- Imprime información sobre las características del sistema

**Resultado**: Objeto configurado listo para entrenar

---

### 2. `focal_loss(self, alpha=0.25, gamma=2.0)`
**Propósito**: Implementa la función de pérdida focal para manejar desbalanceo de clases.

**Parámetros**:
- `alpha`: Factor de peso para clases (0.25 por defecto)
- `gamma`: Factor de enfoque en ejemplos difíciles (2.0 por defecto)

**Qué hace**:
```python
# Fórmula: FL(pt) = -αt(1-pt)^γ * log(pt)
# donde:
# - pt: probabilidad de la clase correcta
# - αt: peso de la clase
# - γ: factor de enfoque
```

**Por qué es importante**:
- La pérdida categórica normal trata todas las clases igual
- Focal Loss penaliza más los errores en clases difíciles/minoritarias
- Reduce la influencia de ejemplos fáciles (bien clasificados)

**Resultado**: Función de pérdida personalizada

---

### 3. `analyze_and_balance_data(self, metadata_path, images_path1, images_path2)`
**Propósito**: Analiza el dataset y prepara los datos para balanceo.

**Parámetros**:
- `metadata_path`: Ruta al CSV con metadatos (HAM10000_metadata.csv)
- `images_path1`: Carpeta con primera parte de imágenes
- `images_path2`: Carpeta con segunda parte de imágenes

**Qué hace**:
1. **Carga metadatos**: Lee el archivo CSV con diagnósticos
2. **Mapea imágenes**: Encuentra todas las imágenes .jpg en ambas carpetas
3. **Filtra datos**: Mantiene solo registros con imágenes existentes
4. **Analiza distribución**: Cuenta cuántas imágenes hay por clase
5. **Identifica problemas**: Detecta clases con menos de 500 muestras
6. **Llama al balanceo**: Invoca `create_balanced_dataset()`

**Salida en consola**:
```
📊 Distribución original de clases:
  akiec:  327 ( 3.3%)
  bcc:    514 ( 5.1%)
  bkl:  1099 (11.0%)
  df:     115 ( 1.1%)
  mel:  1113 (11.1%)
  nv:   6705 (67.0%)
  vasc:   142 ( 1.4%)
```

**Resultado**: DataFrame balanceado + diccionario de rutas de imágenes

---

### 4. `create_balanced_dataset(self, df, class_counts, min_samples=500)`
**Propósito**: Crea un dataset balanceado mediante oversampling.

**Parámetros**:
- `df`: DataFrame con metadatos
- `class_counts`: Conteo de muestras por clase
- `min_samples`: Objetivo mínimo de muestras por clase (500)

**Qué hace**:
1. **Para cada clase**:
   - Si tiene < 500 muestras → **Oversampling**:
     ```python
     # Duplica muestras aleatorias hasta alcanzar min_samples
     n_needed = min_samples - current_count
     extra_samples = df.sample(n=n_needed, replace=True)
     ```
   - Si tiene > 500 muestras → **Undersampling**:
     ```python
     # Selecciona aleatoriamente min_samples
     sampled = df.sample(n=min_samples)
     ```

2. **Concatena todo**: Une todos los DataFrames en uno balanceado

**Técnicas aplicadas**:
- **Oversampling con variación**: Las muestras duplicadas se someterán a augmentation diferente
- **Estratificación**: Mantiene la proporción en train/val/test

**Resultado**: DataFrame con ~500 muestras por clase (~3,500 total)

---

### 5. `create_advanced_generators(self, balanced_df, image_paths, batch_size=32)`
**Propósito**: Crea generadores de datos con augmentation agresivo.

**Parámetros**:
- `balanced_df`: DataFrame balanceado
- `image_paths`: Diccionario con rutas de imágenes
- `batch_size`: Tamaño del lote (28 por defecto = 7 clases × 4 muestras)

**Qué hace**:

#### A. División de datos:
```python
# 70% entrenamiento, 15% validación, 15% prueba
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp
)
```

#### B. Configuración de augmentation (SOLO para entrenamiento):
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normaliza píxeles a [0,1]
    rotation_range=60,           # Rota ±60°
    width_shift_range=0.4,       # Desplaza horizontalmente ±40%
    height_shift_range=0.4,      # Desplaza verticalmente ±40%
    shear_range=0.4,             # Distorsión de corte
    zoom_range=0.5,              # Zoom in/out ±50%
    horizontal_flip=True,        # Voltea horizontalmente
    vertical_flip=True,          # Voltea verticalmente
    brightness_range=[0.5, 1.5], # Ajusta brillo
    channel_shift_range=40,      # Cambia canales RGB
    fill_mode='reflect'          # Rellena bordes reflejando
)
```

#### C. Generador balanceado personalizado:
```python
def balanced_generator():
    """
    Asegura que cada batch contenga igual número de muestras
    de cada clase (4 de cada una = 28 total)
    """
    samples_per_class = batch_size // num_classes  # 28 // 7 = 4
    
    while True:
        batch_images = []
        batch_labels = []
        
        for class_idx in range(7):
            # Selecciona 4 muestras aleatorias de esta clase
            samples = random.sample(class_indices[class_idx], 4)
            
            for sample_idx in samples:
                # Carga imagen, aplica augmentation, normaliza
                img = load_and_augment(image_paths[sample_idx])
                batch_images.append(img)
                batch_labels.append(class_idx)
        
        yield np.array(batch_images), to_categorical(batch_labels, 7)
```

**Por qué es importante el generador balanceado**:
- En cada batch el modelo ve las 7 clases por igual
- Evita que el modelo se sesgue hacia clases mayoritarias
- Mejora el aprendizaje de clases minoritarias

**Resultado**: 3 generadores (train, val, test) + información de pasos

---

### 6. `create_improved_model(self)`
**Propósito**: Construye la arquitectura CNN profunda.

**Arquitectura Detallada**:

```
┌─────────────────────────────────────────────────────────────┐
│                    ARQUITECTURA DEL MODELO                   │
│                     Total: ~23 capas                         │
└─────────────────────────────────────────────────────────────┘

INPUT: 224×224×3 (imagen RGB)
    │
    ▼
┌──────────────────────────────────────┐
│ BLOQUE DE ENTRADA                    │
│ • Conv2D(32, 3×3, relu)             │  → Detecta bordes básicos
│ • BatchNormalization()               │  → Normaliza activaciones
│ • Conv2D(32, 3×3, relu)             │  → Refina características
│ • MaxPooling2D(2×2)                  │  → Reduce a 112×112
│ • Dropout(0.25)                      │  → Previene overfitting
└──────────────────────────────────────┘
    │ Output: 112×112×32
    ▼
┌──────────────────────────────────────┐
│ BLOQUE 1                             │
│ • Conv2D(64, 3×3, relu)             │  → Patrones más complejos
│ • BatchNormalization()               │
│ • Conv2D(64, 3×3, relu)             │
│ • MaxPooling2D(2×2)                  │  → Reduce a 56×56
│ • Dropout(0.25)                      │
└──────────────────────────────────────┘
    │ Output: 56×56×64
    ▼
┌──────────────────────────────────────┐
│ BLOQUE 2 (Más profundo)              │
│ • Conv2D(128, 3×3, relu)            │  → Texturas y patrones
│ • BatchNormalization()               │
│ • Conv2D(128, 3×3, relu)            │
│ • Conv2D(128, 3×3, relu)            │  ← CAPA EXTRA
│ • MaxPooling2D(2×2)                  │  → Reduce a 28×28
│ • Dropout(0.3)                       │
└──────────────────────────────────────┘
    │ Output: 28×28×128
    ▼
┌──────────────────────────────────────┐
│ BLOQUE 3 (Características alto nivel)│
│ • Conv2D(256, 3×3, relu)            │  → Formas complejas
│ • BatchNormalization()               │
│ • Conv2D(256, 3×3, relu)            │
│ • Conv2D(256, 3×3, relu)            │  ← CAPA EXTRA
│ • MaxPooling2D(2×2)                  │  → Reduce a 14×14
│ • Dropout(0.3)                       │
└──────────────────────────────────────┘
    │ Output: 14×14×256
    ▼
┌──────────────────────────────────────┐
│ BLOQUE 4 (Extracción final)          │
│ • Conv2D(512, 3×3, relu)            │  → Características abstractas
│ • BatchNormalization()               │
│ • Conv2D(512, 3×3, relu)            │
│ • GlobalAveragePooling2D()           │  → Reduce a vector 512
│ • Dropout(0.5)                       │
└──────────────────────────────────────┘
    │ Output: 512
    ▼
┌──────────────────────────────────────┐
│ CABEZAL DE CLASIFICACIÓN             │
│ • Dense(1024, relu)                  │  → Capa totalmente conectada
│ • BatchNormalization()               │
│ • Dropout(0.5)                       │
│ • Dense(512, relu)                   │
│ • BatchNormalization()               │
│ • Dropout(0.4)                       │
│ • Dense(256, relu)                   │
│ • BatchNormalization()               │
│ • Dropout(0.3)                       │
│ • Dense(128, relu)                   │
│ • Dropout(0.2)                       │
│ • Dense(7, softmax)                  │  → 7 probabilidades (salida)
└──────────────────────────────────────┘
    │
    ▼
OUTPUT: [p₁, p₂, p₃, p₄, p₅, p₆, p₇]
        Probabilidades para cada clase
```

**Técnicas aplicadas**:

1. **BatchNormalization**:
   - Normaliza activaciones entre capas
   - Acelera entrenamiento
   - Reduce overfitting

2. **Dropout**:
   - Desactiva aleatoriamente neuronas durante entrenamiento
   - Previene co-adaptación de características
   - Aumentos graduales: 0.25 → 0.3 → 0.5

3. **GlobalAveragePooling2D**:
   - Reduce dimensionalidad sin perder información
   - Más robusto que Flatten
   - Reduce parámetros

4. **Capas extra en bloques 2 y 3**:
   - Aumenta capacidad de aprendizaje
   - Captura patrones más complejos

**Parámetros totales**: ~15-20 millones (depende de implementación exacta)

**Resultado**: Modelo CNN compilado

---

### 7. `compile_model_with_focal_loss(self, learning_rate=0.001)`
**Propósito**: Configura el optimizador y la función de pérdida.

**Parámetros**:
- `learning_rate`: Tasa de aprendizaje inicial (0.001)

**Qué hace**:
```python
optimizer = Adam(
    learning_rate=0.001,  # Tasa de aprendizaje
    beta_1=0.9,           # Momento exponencial
    beta_2=0.999,         # Momento exponencial al cuadrado
    epsilon=1e-07         # Estabilidad numérica
)

model.compile(
    optimizer=optimizer,
    loss=focal_loss,      # Pérdida personalizada
    metrics=['accuracy', 'precision', 'recall']
)
```

**Métricas monitoreadas**:
- **Accuracy**: % de predicciones correctas
- **Precision**: De las predicciones positivas, cuántas son correctas
- **Recall**: De los casos positivos reales, cuántos detecta

**Resultado**: Modelo compilado listo para entrenar

---

### 8. `create_callbacks(self, model_save_path)`
**Propósito**: Configura callbacks para controlar el entrenamiento.

**Callbacks implementados**:

#### A. **EarlyStopping**
```python
EarlyStopping(
    monitor='val_accuracy',      # Métrica a vigilar
    patience=25,                 # Espera 25 épocas sin mejora
    restore_best_weights=True,   # Restaura mejores pesos
    mode='max'                   # Maximizar accuracy
)
```
- **Qué hace**: Detiene entrenamiento si no mejora en 25 épocas
- **Por qué**: Evita overfitting y ahorra tiempo

#### B. **ModelCheckpoint**
```python
ModelCheckpoint(
    'models/improved_balanced_7class_model.h5',
    monitor='val_accuracy',
    save_best_only=True,         # Solo guarda si mejora
    save_weights_only=False      # Guarda modelo completo
)
```
- **Qué hace**: Guarda el mejor modelo durante entrenamiento
- **Por qué**: Preserva el mejor modelo aunque luego empeore

#### C. **ReduceLROnPlateau**
```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,                  # Reduce lr a 20% (×0.2)
    patience=12,                 # Espera 12 épocas
    min_lr=1e-8                  # Límite inferior
)
```
- **Qué hace**: Reduce learning rate cuando se estanca
- **Por qué**: Ayuda a escapar de mínimos locales
- **Ejemplo**: 0.001 → 0.0002 → 0.00004 → ...

**Resultado**: Lista de callbacks para fit()

---

### 9. `train_improved_model(self, train_gen, val_gen, steps_per_epoch, val_steps, epochs=80, model_save_path=None)`
**Propósito**: Ejecuta el entrenamiento completo del modelo.

**Parámetros**:
- `train_gen`: Generador de datos de entrenamiento
- `val_gen`: Generador de validación
- `steps_per_epoch`: Pasos por época (~70-100)
- `val_steps`: Pasos de validación (~15-20)
- `epochs`: Número de épocas (80)
- `model_save_path`: Ruta para guardar modelo

**Qué hace**:
```python
history = model.fit(
    train_gen,                    # Datos de entrenamiento
    steps_per_epoch=70,           # 70 batches por época
    epochs=80,                    # 80 épocas completas
    validation_data=val_gen,      # Datos de validación
    validation_steps=15,          # 15 batches de validación
    callbacks=[early_stop, checkpoint, reduce_lr]
)
```

**Proceso en cada época**:
1. **Entrenamiento**:
   - Procesa 70 batches de 28 imágenes (1,960 imágenes)
   - Calcula pérdida y actualiza pesos
   - Registra accuracy, precision, recall

2. **Validación**:
   - Evalúa en 15 batches sin actualizar pesos
   - Verifica si hay overfitting
   - Decide si guardar modelo o reducir lr

3. **Callbacks**:
   - Verifica si debe parar early
   - Guarda modelo si mejora val_accuracy
   - Reduce lr si se estanca val_loss

**Tiempo estimado**: 3-5 horas (depende del hardware)

**Resultado**: Objeto History con métricas de todas las épocas

---

### 10. `evaluate_improved_model(self, test_gen, test_steps, data_splits)`
**Propósito**: Evalúa el modelo en el conjunto de prueba.

**Qué hace**:

#### A. Evaluación general:
```python
test_results = model.evaluate(test_gen, steps=test_steps)
# Retorna: [loss, accuracy, precision, recall]
```

#### B. Predicciones individuales:
```python
predictions = model.predict(test_gen, steps=test_steps)
# Para cada imagen: [p_akiec, p_bcc, p_bkl, p_df, p_mel, p_nv, p_vasc]

predicted_classes = np.argmax(predictions, axis=1)
# Clase con mayor probabilidad: [0-6]
```

#### C. Reporte de clasificación:
```python
from sklearn.metrics import classification_report

report = classification_report(
    true_classes,           # Etiquetas reales
    predicted_classes,      # Predicciones
    target_names=[...],     # Nombres de clases
    output_dict=True
)
```

**Métricas por clase**:
```
              precision    recall  f1-score   support

       akiec       0.75      0.82      0.78        75
         bcc       0.88      0.85      0.86        77
         bkl       0.82      0.79      0.80        75
          df       0.91      0.88      0.89        75
         mel       0.84      0.87      0.85        75
          nv       0.93      0.91      0.92        75
        vasc       0.87      0.89      0.88        75

    accuracy                           0.86       527
   macro avg       0.86      0.86      0.86       527
weighted avg       0.86      0.86      0.86       527
```

#### D. Matriz de confusión:
```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(true_classes, predicted_classes)
```

**Ejemplo de matriz de confusión**:
```
           akiec  bcc  bkl   df  mel   nv  vasc
akiec        62    2    5    0    4    2     0
bcc           1   65    3    0    2    6     0
bkl           3    2   59    0    5    6     0
df            0    0    1   66    2    5     1
mel           2    1    4    1   65    2     0
nv            1    3    4    2    2   68     0
vasc          0    0    1    1    1    0    72
```
- Diagonal: predicciones correctas
- Fuera de diagonal: confusiones

**Resultado**: Diccionario con todas las métricas y predicciones

---

### 11. `plot_improved_results(self, evaluation_results)`
**Propósito**: Genera visualizaciones de los resultados.

**Gráficas generadas**:

#### A. Historial de entrenamiento (4 subplots):
```python
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Accuracy
plt.plot(history['accuracy'], label='Train')
plt.plot(history['val_accuracy'], label='Validation')

# 2. Loss
plt.plot(history['loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')

# 3. Precision
plt.plot(history['precision'], label='Train')
plt.plot(history['val_precision'], label='Validation')

# 4. Recall
plt.plot(history['recall'], label='Train')
plt.plot(history['val_recall'], label='Validation')
```

**Qué buscar**:
- ✅ Curvas de train y val cercanas → Buen generalización
- ❌ Gran separación → Overfitting
- ✅ Curva de loss descendente → Aprendiendo
- ❌ Loss oscilante → Learning rate alto

#### B. Matriz de confusión normalizada:
```python
cm_normalized = cm / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_normalized, annot=True, fmt='.2f',
            xticklabels=class_names,
            yticklabels=class_names,
            cmap='Blues')
```

**Interpretación**:
- Valores altos en diagonal → Buenas predicciones
- Valores altos fuera de diagonal → Confusiones frecuentes

**Archivos guardados**:
- `evaluation/improved_7class_training_history.png`
- `evaluation/improved_7class_confusion_matrix.png`

**Resultado**: Visualizaciones guardadas en carpeta evaluation/

---

### 12. `convert_to_tflite(self, model_path, output_path)`
**Propósito**: Convierte el modelo .h5 a TensorFlow Lite para móviles.

**Parámetros**:
- `model_path`: Ruta del modelo .h5 entrenado
- `output_path`: Ruta para guardar .tflite

**Qué hace**:

#### A. Carga el modelo:
```python
model = tf.keras.models.load_model(
    model_path,
    compile=False  # No necesita compilar para conversión
)
```

#### B. Configuración del convertidor:
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimizaciones
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Incluye:
# - Cuantización de pesos (float32 → int8)
# - Fusión de operaciones
# - Eliminación de operaciones redundantes
```

#### C. Conversión:
```python
tflite_model = converter.convert()

# Guarda el archivo
with open(output_path, 'wb') as f:
    f.write(tflite_model)
```

#### D. Compresión típica:
```
Original .h5:     85.4 MB
Convertido .tflite: 21.8 MB
Compresión:        3.9x
```

**Ventajas de TFLite**:
- 📦 Tamaño reducido (3-5x más pequeño)
- ⚡ Inferencia más rápida en móviles
- 🔋 Menor consumo de batería
- 📱 Optimizado para Android/iOS
- 🚫 No requiere TensorFlow completo

**Resultado**: Archivo .tflite listo para deployment en apps móviles

---

## 🚀 Proceso de Entrenamiento Completo

### Función `main()` - Pipeline Completo

```python
def main():
    # 1. CONFIGURACIÓN
    METADATA_PATH = 'datasets/ham10000/HAM10000_metadata.csv'
    IMAGES_PATH1 = 'datasets/ham10000/HAM10000_images_part_1'
    IMAGES_PATH2 = 'datasets/ham10000/HAM10000_images_part_2'
    IMG_SIZE = 224
    BATCH_SIZE = 28  # 7 clases × 4 muestras
    EPOCHS = 80
    
    # 2. CREAR ENTRENADOR
    trainer = ImprovedBalanced7ClassModel(img_size=IMG_SIZE)
    
    # 3. ANÁLISIS Y BALANCEO DE DATOS
    balanced_df, image_paths = trainer.analyze_and_balance_data(
        METADATA_PATH, IMAGES_PATH1, IMAGES_PATH2
    )
    # Output: ~3,500 muestras balanceadas
    
    # 4. CREAR GENERADORES
    train_gen, val_gen, test_gen, steps_per_epoch, val_steps, test_steps, data_splits = \
        trainer.create_advanced_generators(balanced_df, image_paths, BATCH_SIZE)
    # Train: ~2,450 | Val: ~525 | Test: ~525
    
    # 5. CONSTRUIR MODELO
    model = trainer.create_improved_model()
    # 23 capas, ~15M parámetros
    
    # 6. COMPILAR CON FOCAL LOSS
    trainer.compile_model_with_focal_loss(learning_rate=0.001)
    
    # 7. ENTRENAR
    history = trainer.train_improved_model(
        train_gen, val_gen, steps_per_epoch, val_steps,
        epochs=EPOCHS,
        model_save_path='models/improved_balanced_7class_model.h5'
    )
    # Duración: 3-5 horas
    
    # 8. EVALUAR
    evaluation_results = trainer.evaluate_improved_model(
        test_gen, test_steps, data_splits
    )
    # Accuracy objetivo: >85%
    
    # 9. VISUALIZAR
    trainer.plot_improved_results(evaluation_results)
    # Guarda gráficas en evaluation/
    
    # 10. CONVERTIR A TFLITE
    tflite_path = trainer.convert_to_tflite(
        'models/improved_balanced_7class_model.h5',
        'models/flutter_assets/improved_balanced_7class_model.tflite'
    )
    # Reduce tamaño ~4x
    
    # 11. RESUMEN FINAL
    print(f"🎉 Entrenamiento completado!")
    print(f"📊 Test Accuracy: {evaluation_results['test_results'][1]:.4f}")
    print(f"📱 Modelo TFLite: {tflite_path}")
```

---

## 💻 Tecnologías Utilizadas

### Librerías principales:

| Librería | Versión | Propósito |
|----------|---------|-----------|
| **TensorFlow** | 2.15.0+ | Framework de Deep Learning |
| **Keras** | 2.15.0+ | API de alto nivel para redes neuronales |
| **NumPy** | 1.24.0+ | Operaciones numéricas y arrays |
| **Pandas** | 2.0.0+ | Manipulación de datos y DataFrames |
| **Scikit-learn** | 1.3.0+ | Métricas y división de datos |
| **Matplotlib** | 3.7.0+ | Visualización de gráficas |
| **Seaborn** | 0.12.0+ | Visualización estadística |
| **Pillow** | 10.0.0+ | Procesamiento de imágenes |

### Algoritmos y técnicas:

- **CNN (Convolutional Neural Networks)**: Extracción de características visuales
- **Focal Loss**: Manejo de desbalanceo de clases
- **Data Augmentation**: Aumento artificial del dataset
- **Batch Normalization**: Normalización entre capas
- **Dropout**: Regularización para prevenir overfitting
- **Adam Optimizer**: Optimizador adaptativo
- **Early Stopping**: Prevención de sobreentrenamiento
- **Learning Rate Scheduling**: Ajuste dinámico del lr
- **Stratified Splitting**: División balanceada de datos
- **Oversampling**: Balanceo de clases minoritarias

---

## 📋 Requisitos

### Archivos necesarios:

```
ai-model/
├── improved_balanced_7class_training.py  (este script)
├── requirements.txt
└── datasets/
    └── ham10000/
        ├── HAM10000_metadata.csv
        ├── HAM10000_images_part_1/
        │   └── *.jpg (5,000 imágenes)
        └── HAM10000_images_part_2/
            └── *.jpg (5,015 imágenes)
```

### Dataset HAM10000:
- **Nombre completo**: Human Against Machine with 10000 training images
- **Fuente**: Harvard Dataverse
- **Tamaño**: 10,015 imágenes dermoscópicas
- **Formato**: JPG, RGB, varios tamaños
- **Clases**: 7 tipos de lesiones cutáneas
- **Licencia**: CC BY-NC 4.0

### Hardware recomendado:
- **GPU**: NVIDIA con CUDA (RTX 3060 o superior)
- **RAM**: 16 GB mínimo
- **Almacenamiento**: 10 GB libres
- **CPU**: Multi-core (i7/Ryzen 7 o superior)

**Tiempo de entrenamiento**:
- Con GPU: 2-3 horas
- Sin GPU (solo CPU): 15-24 horas ⚠️

---

## 🎯 Uso

### 1. Instalar dependencias:
```bash
cd ai-model
pip install -r requirements.txt
```

### 2. Descargar dataset HAM10000:
```bash
# Opción 1: Descarga manual desde Harvard Dataverse
# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

# Opción 2: Usar script (si disponible)
python download_dataset.py
```

### 3. Ejecutar entrenamiento:
```bash
python improved_balanced_7class_training.py
```

### 4. Monitorear progreso:
```
🧬 Modelo mejorado y equilibrado de 7 clases de enfermedades de la piel
======================================================================
🔍 Análisis y Balanceo de Datos...
📊 Distribución original de clases:
  akiec:  327 ( 3.3%)
  bcc:    514 ( 5.1%)
  ...
⚖️ Balanceo de datos (objetivo: min 500 muestras/clases)...
✅ Conjunto de datos equilibrado:
  akiec: 500 ejemplo
  bcc: 500 ejemplo
  ...
🔄 Creando generadores avanzados...
🏗️ Se está creando el modelo mejorado...
⚙️ Compilando el modelo Focal Loss...
🚀 Entrenamiento del Modelo Mejorado Comenzando... (80 epoch)

Epoch 1/80
70/70 [==============================] - 145s 2s/step - loss: 1.9234 - accuracy: 0.3254 - val_loss: 1.6543 - val_accuracy: 0.4123
...
```

### 5. Resultados:
```
models/
├── improved_balanced_7class_model.h5        (modelo completo)
└── flutter_assets/
    └── improved_balanced_7class_model.tflite (modelo optimizado)

evaluation/
├── improved_7class_training_history.png
└── improved_7class_confusion_matrix.png
```

---

## 📊 Resultados Esperados

### Métricas objetivo:

| Métrica | Modelo Anterior | Modelo Mejorado | Mejora |
|---------|-----------------|-----------------|--------|
| **Accuracy global** | ~70% | >85% | +15% |
| **akiec (precision)** | 0.45 | >0.75 | +67% |
| **bcc (precision)** | 0.62 | >0.85 | +37% |
| **df (precision)** | 0.38 | >0.88 | +132% |
| **vasc (precision)** | 0.51 | >0.87 | +71% |
| **Clases fuertes (nv, mel, bkl)** | ~85% | >90% | +5% |

### Mejoras implementadas:

✅ **Focal Loss**: Resuelve desbalanceo de clases
✅ **Muestreo equilibrado**: Todas las clases tienen igual representación
✅ **Augmentation agresivo**: Mayor variabilidad de datos
✅ **Arquitectura más profunda**: Mejor extracción de características
✅ **Callbacks inteligentes**: Previene overfitting y optimiza lr
✅ **Optimización TFLite**: Modelo 4x más pequeño y rápido

---

## 🔍 Solución de Problemas

### Problema: "Out of Memory" (GPU)
**Solución**:
```python
# Reduce batch_size
BATCH_SIZE = 14  # En lugar de 28

# O reduce img_size
IMG_SIZE = 128  # En lugar de 224
```

### Problema: "FileNotFoundError: datasets/..."
**Solución**:
```bash
# Verifica que el dataset esté en la ruta correcta
ls datasets/ham10000/
# Debe mostrar: HAM10000_metadata.csv, HAM10000_images_part_1/, HAM10000_images_part_2/
```

### Problema: Entrenamiento muy lento sin GPU
**Solución**:
```python
# Reduce épocas y tamaño de imagen
EPOCHS = 30
IMG_SIZE = 128
BATCH_SIZE = 16
```

### Problema: Val_accuracy no mejora después de muchas épocas
**Solución**:
```python
# Ajusta learning rate y patience
trainer.compile_model_with_focal_loss(learning_rate=0.0001)

# Modifica early stopping
EarlyStopping(patience=15)  # Menos paciencia
```

---

## 📚 Referencias

### Papers científicos:
1. **Focal Loss**: [Lin et al. 2017 - "Focal Loss for Dense Object Detection"](https://arxiv.org/abs/1708.02002)
2. **HAM10000 Dataset**: [Tschandl et al. 2018 - "The HAM10000 dataset"](https://doi.org/10.1038/sdata.2018.161)
3. **Data Augmentation**: [Shorten & Khoshgoftaar 2019](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0)
4. **BatchNormalization**: [Ioffe & Szegedy 2015](https://arxiv.org/abs/1502.03167)

### Recursos:
- TensorFlow Documentation: https://www.tensorflow.org/
- Keras API: https://keras.io/
- HAM10000 Dataset: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

---

## 🎓 Conceptos Clave

### ¿Qué es Focal Loss?
La pérdida focal es una modificación de la pérdida de entropía cruzada que reduce la influencia de ejemplos fáciles y se enfoca en ejemplos difíciles. Es especialmente útil para datasets desbalanceados.

**Fórmula**:
```
FL(pt) = -αt(1-pt)^γ * log(pt)

donde:
- pt: probabilidad de la clase correcta
- αt: peso de balanceo (0.25)
- γ: factor de enfoque (2.0)
```

**Ejemplo**:
- Ejemplo fácil (pt=0.9): (1-0.9)^2 = 0.01 → peso bajo
- Ejemplo difícil (pt=0.3): (1-0.3)^2 = 0.49 → peso alto

### ¿Qué es Data Augmentation?
Técnica que crea variaciones artificiales de imágenes para aumentar el tamaño del dataset y mejorar la generalización del modelo.

**Transformaciones aplicadas**:
- **Geométricas**: rotación, desplazamiento, zoom, flip
- **Fotométricas**: brillo, contraste, saturación
- **Espaciales**: distorsión, corte

### ¿Qué es Overfitting?
Fenómeno donde el modelo memoriza los datos de entrenamiento pero no generaliza bien a datos nuevos.

**Indicadores**:
- Train accuracy >> Val accuracy
- Val loss aumenta mientras train loss disminuye

**Prevención**:
- Dropout
- Data Augmentation
- Early Stopping
- Regularización

---

## 👨‍💻 Autor

Proyecto desarrollado para detección de enfermedades de la piel usando Deep Learning.

---

## 📄 Licencia

Este código está diseñado para propósitos educativos y de investigación.

⚠️ **Advertencia médica**: Este modelo es solo para demostración y no debe usarse para diagnósticos médicos reales sin supervisión profesional.

---

## 🚀 Próximos Pasos

1. ✅ Entrenar modelo con focal loss y balanceo
2. ✅ Convertir a TFLite para móviles
3. 📱 Integrar en aplicación Django
4. 🔄 Implementar feedback loop para mejorar continuamente
5. 🌐 Desplegar en producción con API REST

---

**¡Listo para entrenar! 🚀**

Para cualquier duda, revisa la sección de solución de problemas o consulta la documentación de TensorFlow/Keras.
