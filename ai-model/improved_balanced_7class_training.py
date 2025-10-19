#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import warnings
warnings.filterwarnings('ignore')

class ImprovedBalanced7ClassModel:
    def __init__(self, img_size=224):
        self.img_size = img_size
        self.num_classes = 7
        self.model = None
        self.history = None
        
        # 7-sınıf tanımları
        self.class_names = {
            'akiec': 'Actinic keratoses', #Queratosis actínicas',      
            'bcc': 'Basal cell carcinoma',  #Carcinoma basal celular',   
            'bkl': 'Benign keratosis',   #  Queratosis benigna',
            'df': 'Dermatofibroma',      #'Dermatofibroma',       
            'mel': 'Melanoma',            #'Melanoma',   
            'nv': 'Melanocytic nevi',     #Nevos melanocíticos',      
            'vasc': 'Vascular lesions'     #'Lesiones vasculares'
        }
        
        self.class_list = list(self.class_names.keys())
        
        print("🧬 Modelo mejorado y equilibrado de 7 clases de enfermedades de la piel")
        print("=" * 70)
        print("🎯 Características:")
        print("  ✅ Pérdida focal (para desequilibrio de datos)")
        print("  ✅ Muestreo equilibrado")
        print("  ✅ Aumento agresivo de datos")
        print("  ✅ Arquitectura avanzada")
        print("  ✅ Generación de datos sintéticos")
    
    def focal_loss(self, alpha=0.25, gamma=2.0):
        """Pérdida focal: por desequilibrio de datos"""
        def focal_loss_fixed(y_true, y_pred):
            epsilon = K.epsilon()
            y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
            
            # Convierta a float32 para evitar problemas de tipo
            y_true = K.cast(y_true, tf.float32)
            y_pred = K.cast(y_pred, tf.float32)
            
            # Calcular la pérdida focal
            alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)
            p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_pred) - y_pred)
            focal_loss = - alpha_t * K.pow((K.ones_like(p_t) - p_t), gamma) * K.log(p_t)
            
            return K.mean(K.sum(focal_loss, axis=-1))
        
        return focal_loss_fixed
    
    def analyze_and_balance_data(self, metadata_path, images_path1, images_path2):
        """Análisis y balanceo de datos"""
        print("\n🔍 Análisis y Balanceo de Datos...")
        
        # Subir metadatos
        df = pd.read_csv(metadata_path)
        
        # Rutas de imagen
        image_paths = {}
        for img_path in [images_path1, images_path2]:
            if os.path.exists(img_path):
                for img_file in os.listdir(img_path):
                    if img_file.endswith('.jpg'):
                        image_id = img_file.replace('.jpg', '')
                        image_paths[image_id] = os.path.join(img_path, img_file)
        
        df_filtered = df[df['image_id'].isin(image_paths.keys())].copy()
        
        # Análisis de distribución de clases
        class_counts = df_filtered['dx'].value_counts()
        print(f"📊 Distribución original de clases:")
        for class_code in self.class_list:
            count = class_counts.get(class_code, 0)
            percentage = (count / len(df_filtered)) * 100
            print(f"  {class_code}: {count:4d} (%{percentage:5.1f})")
        
        # Identificar clases de problemas
        min_samples = 500  # Objetivo mínimo
        problem_classes = []
        
        for class_code in self.class_list:
            count = class_counts.get(class_code, 0)
            if count < min_samples:
                problem_classes.append(class_code)
                print(f"⚠️ {class_code}: {count} < {min_samples} (clase de problemas)")
        
        # Estrategia de muestreo balanceado
        balanced_df = self.create_balanced_dataset(df_filtered, class_counts, min_samples)
        
        return balanced_df, image_paths
    
    def create_balanced_dataset(self, df, class_counts, min_samples=500):
        """Crear un conjunto de datos equilibrado"""
        print(f"\n⚖️ Balanceo de datos (objetivo: min {min_samples} muestras/clases)...")
        
        balanced_frames = []
        
        for class_code in self.class_list:
            class_df = df[df['dx'] == class_code].copy()
            current_count = len(class_df)
            
            if current_count < min_samples:
                # Haremos upsampling - aumento sintético
                needed = min_samples - current_count
                print(f"  📈 {class_code}: {current_count} → {min_samples} (+{needed} synthetic)")
                
                # Repetir muestras existentes (recibirán un aumento diferente)
                repeats = (needed // current_count) + 1
                extended_df = pd.concat([class_df] * (repeats + 1), ignore_index=True)
                extended_df = extended_df.iloc[:min_samples]  # Obtener exactamente min_samples
                
            else:
                # Submuestreo: reducir muestras redundantes
                target_count = min(current_count, min_samples * 2)  
                extended_df = class_df.sample(n=target_count, random_state=42)
                print(f"  📉 {class_code}: {current_count} → {target_count}")
            
            balanced_frames.append(extended_df)
        
        balanced_df = pd.concat(balanced_frames, ignore_index=True)
        
        # Análisis de resultados
        print(f"\n✅ Conjunto de datos equilibrado:")
        new_counts = balanced_df['dx'].value_counts()
        for class_code in self.class_list:
            count = new_counts.get(class_code, 0)
            print(f"  {class_code}: {count} ejemplo")
        
        print(f"📊 Total: {len(balanced_df)} ejemplo")
        
        return balanced_df
    
    def create_advanced_generators(self, balanced_df, image_paths, batch_size=32):
        """Generadores avanzados: muestreo igual para cada clase"""
        print(f"\n🔄 Creando generadores avanzados...")
        
        # Agregar etiquetas de clase
        balanced_df['class_idx'] = balanced_df['dx'].map(
            {class_code: idx for idx, class_code in enumerate(self.class_list)}
        )
        
        # División de tren-val-prueba
        X = balanced_df['image_id'].values
        y = balanced_df['class_idx'].values
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )
        
        print(f"📚 Educación: {len(X_train)} ejemplo")
        print(f"📊 Validación: {len(X_val)} ejemplo")
        print(f"🧪 Test: {len(X_test)} ejemplo")

        # Muy agresivo aumento
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=60,          # Más rotación
            width_shift_range=0.4,      # Más desplazamiento
            height_shift_range=0.4,
            shear_range=0.4,            # Más flexión
            zoom_range=0.5,             # Más zoom
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.5, 1.5], # Brillo más amplio
            channel_shift_range=40,      # Más cambios de color
            fill_mode='reflect',
            # Un 'lar de aumento'
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False
        )
        
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Función de generador equilibrado
        def balanced_generator(image_ids, labels, datagen, batch_size, shuffle=True):
            """Muestras iguales de cada clase en cada lote"""
            
            # Agrupar por clases
            class_indices = {}
            for i, label in enumerate(labels):
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(i)
            
            samples_per_class = batch_size // self.num_classes
            
            while True:
                batch_x = []
                batch_y = []
                
                # Tome muestras iguales de cada clase
                for class_idx in range(self.num_classes):
                    if class_idx in class_indices:
                        indices = class_indices[class_idx]
                        if shuffle:
                            selected_indices = np.random.choice(indices, samples_per_class, replace=True)
                        else:
                            selected_indices = indices[:samples_per_class]
                        
                        for idx in selected_indices:
                            image_id = image_ids[idx]
                            label = labels[idx]
                            
                            if image_id in image_paths:
                                try:
                                    img = load_img(image_paths[image_id], 
                                                 target_size=(self.img_size, self.img_size))
                                    img_array = img_to_array(img)
                                    
                                    # Aplicar aumento
                                    if datagen is not None:
                                        img_array = datagen.random_transform(img_array)
                                        img_array = datagen.standardize(img_array)
                                    else:
                                        img_array = img_array / 255.0
                                    
                                    batch_x.append(img_array)
                                    batch_y.append(label)
                                    
                                except Exception as e:
                                    continue
                
                if len(batch_x) > 0:
                    # Mezclar el lote
                    if shuffle:
                        indices = np.arange(len(batch_x))
                        np.random.shuffle(indices)
                        batch_x = [batch_x[i] for i in indices]
                        batch_y = [batch_y[i] for i in indices]
                    
                    X = np.array(batch_x)
                    y = to_categorical(np.array(batch_y), num_classes=self.num_classes)
                    yield X, y
        
        # Crear generadores
        train_gen = balanced_generator(X_train, y_train, train_datagen, batch_size, True)
        val_gen = balanced_generator(X_val, y_val, val_test_datagen, batch_size, False)
        test_gen = balanced_generator(X_test, y_test, val_test_datagen, batch_size, False)
        
        # Pasos Haspla
        steps_per_epoch = len(X_train) // batch_size
        val_steps = len(X_val) // batch_size
        test_steps = len(X_test) // batch_size
        
        print(f"✅ Generadores balanceados listos!")
        print(f"📚 Pasos por época: {steps_per_epoch}")
        print(f"📊 Pasos de validación: {val_steps}")
        print(f"🧪 Pasos de prueba: {test_steps}")

        data_splits = {
            'train': {'images': X_train, 'labels': y_train},
            'val': {'images': X_val, 'labels': y_val},
            'test': {'images': X_test, 'labels': y_test}
        }
        
        return train_gen, val_gen, test_gen, steps_per_epoch, val_steps, test_steps, data_splits
    
    def create_improved_model(self):
        """Arquitectura de modelo mejorada"""
        print(f"\n🏗️ Se está creando el modelo mejorado...")
        
        model = Sequential([
            # Capas de entrada: filtros más pequeños
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size, self.img_size, 3)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Bloque 1 - Extracción de características
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Bloque 2 - Funciones más profundas
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            Conv2D(128, (3, 3), activation='relu'),  # Capa extra
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            
            # Bloque 3 - Características de alto nivel
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu'),
            Conv2D(256, (3, 3), activation='relu'),  # Capa extra
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            
            # Bloque 4 - Extracción de características finales
            Conv2D(512, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(512, (3, 3), activation='relu'),
            GlobalAveragePooling2D(),
            Dropout(0.5),
            
            # Cabezal de clasificación - más grande
            Dense(1024, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            
            # Output layer
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        
        print(f"✅ ¡Modelo mejorado creado!")
        print(f"📊 Parámetros totales: {self.model.count_params():,}")
        print(f"🏗️ Número de capas: {len(self.model.layers)}")
        
        return self.model
    
    def compile_model_with_focal_loss(self, learning_rate=0.001):
        """Compilar modelo con Focal Loss"""
        print("\n⚙️ Compilando el modelo Focal Loss...")
        
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        # Usar Focal loss
        focal_loss_fn = self.focal_loss(alpha=0.25, gamma=2.0)
        
        self.model.compile(
            optimizer=optimizer,
            loss=focal_loss_fn,
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("✅ Compilación del modelo Focal Loss completada!")
        print("🎯 Focal Loss para solucionar problemas de desbalanceo")
    
    def create_callbacks(self, model_save_path):
        """Crear callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=25,  # Más paciencia
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            
            ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                mode='max'
            ),
            
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,  # Descenso más agresivo
                patience=12,
                min_lr=1e-8,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_improved_model(self, train_gen, val_gen, steps_per_epoch, val_steps,
                           epochs=80, model_save_path=None):
        """Entrenamiento del modelo mejorado"""
        if model_save_path is None:
            model_save_path = 'models/improved_balanced_7class_model.h5'
        
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        print(f"\n🚀 Entrenamiento del Modelo Mejorado Comenzando... ({epochs} epoch)")
        print("🎯 Con Focal Loss + Muestreo Balanceado")
        print("Este proceso puede tardar de 3 a 5 horas!")

        callbacks = self.create_callbacks(model_save_path)
        
        self.history = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Entrenamiento del modelo mejorado completado!")
        return self.history
    
    def evaluate_improved_model(self, test_gen, test_steps, data_splits):
        """Evaluación mejorada del modelo"""
        print("\n🧪 Evaluación del modelo mejorado...")
        
        test_results = self.model.evaluate(test_gen, steps=test_steps, verbose=0)
        
        print(f"\n📊 Resultados de la prueba:")
        metrics = ['loss', 'accuracy', 'precision', 'recall']
        for i, metric in enumerate(metrics):
            if i < len(test_results):
                print(f"  {metric.capitalize()}: {test_results[i]:.4f}")
        
        # Predictions
        predictions = self.model.predict(test_gen, steps=test_steps, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # True labels
        true_classes = data_splits['test']['labels'][:len(predicted_classes)]
        
        print(f"🔍 Tamaño de la predicción: {len(predicted_classes)}")
        print(f"🔍 Tamaño de las etiquetas verdaderas: {len(true_classes)}")
        
        # Informe de clasificación
        class_names_list = [self.class_names[code] for code in self.class_list]
        
        try:
            report = classification_report(
                true_classes, predicted_classes,
                target_names=class_names_list,
                output_dict=True,
                zero_division=0
            )
            
            cm = confusion_matrix(true_classes, predicted_classes)
            
            print(f"\n📋 Resultados mejorados clase por clase:")
            for i, class_code in enumerate(self.class_list):
                class_name = self.class_names[class_code]
                if class_name in report:
                    metrics_data = report[class_name]
                    print(f"  {class_code} ({class_name}):")
                    print(f"    Precision: {metrics_data['precision']:.4f}")
                    print(f"    Recall: {metrics_data['recall']:.4f}")
                    print(f"    F1-Score: {metrics_data['f1-score']:.4f}")
                    print(f"    Support: {metrics_data['support']}")
        
        except Exception as e:
            print(f"⚠️ Error en el informe de clasificación: {e}")
            report = {}
            cm = np.zeros((self.num_classes, self.num_classes))
        
        return {
            'test_results': test_results,
            'predictions': predictions,
            'predicted_classes': predicted_classes,
            'true_classes': true_classes,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def plot_improved_results(self, evaluation_results):
        """Visualizar resultados mejorados"""
        print("\n📊 Visualizando resultados mejorados...")
        
        os.makedirs('evaluation', exist_ok=True)
        
        # Training History
        if self.history:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            metrics = ['accuracy', 'loss', 'precision', 'recall']
            titles = ['Accuracy', 'Loss', 'Precision', 'Recall']
            
            for i, (metric, title) in enumerate(zip(metrics, titles)):
                row = i // 2
                col = i % 2
                
                if metric in self.history.history:
                    axes[row, col].plot(self.history.history[metric], label='Training', linewidth=2)
                    if f'val_{metric}' in self.history.history:
                        axes[row, col].plot(self.history.history[f'val_{metric}'], label='Validation', linewidth=2)
                    axes[row, col].set_title(f'Improved Model {title}', fontsize=14, fontweight='bold')
                    axes[row, col].set_xlabel('Epoch')
                    axes[row, col].set_ylabel(title)
                    axes[row, col].legend()
                    axes[row, col].grid(True, alpha=0.3)

            plt.suptitle('Modelo mejorado - Historial de entrenamiento', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('evaluation/improved_7class_training_history.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Confusion Matrix
        if len(evaluation_results['confusion_matrix']) > 0:
            cm = evaluation_results['confusion_matrix']
            plt.figure(figsize=(12, 10))
            
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            class_labels = [f"{code}\\n{self.class_names[code]}" for code in self.class_list]
            
            sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                       xticklabels=class_labels, yticklabels=class_labels,
                       cbar_kws={'label': 'Normalized Frequency'})
            plt.title('Mejorado Matriz de confusión normalizada de clasificación de enfermedades de la piel de 7 clases', 
                     fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig('evaluation/improved_7class_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print("📊¡Se han registrado visualizaciones mejoradas!")
    
    def convert_to_tflite(self, model_path, output_path):
        """Conversión de TFLite"""
        print(f"\n📱 Conversión de modelo TFLite mejorada...")
        
        model = tf.keras.models.load_model(model_path, compile=False)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        try:
            converter.target_spec.supported_types = [tf.float16]
            tflite_model = converter.convert()
        except:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        original_size = os.path.getsize(model_path) / (1024 * 1024)
        tflite_size = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"✅ ¡El modelo TFLite mejorado está listo!")
        print(f"📊 Original: {original_size:.1f} MB → TFLite: {tflite_size:.1f} MB")
        print(f"🗜️ Compresión: {original_size/tflite_size:.1f}x")

        return output_path

def main():
    print("🧬 Modelo mejorado y equilibrado de enfermedades de la piel de 7 clases")
    print("=" * 80)
    
    # Parámetros
    METADATA_PATH = 'datasets/ham10000/HAM10000_metadata.csv'
    IMAGES_PATH1 = 'datasets/ham10000/HAM10000_images_part_1'
    IMAGES_PATH2 = 'datasets/ham10000/HAM10000_images_part_2'
    
    IMG_SIZE = 224
    BATCH_SIZE = 28  #7 clases * 4 muestras = 28 (balanceadas)
    EPOCHS = 80
    
    # Crear un entrenador de modelos
    trainer = ImprovedBalanced7ClassModel(img_size=IMG_SIZE)
    
    # Análisis y balanceo de datos
    balanced_df, image_paths = trainer.analyze_and_balance_data(
        METADATA_PATH, IMAGES_PATH1, IMAGES_PATH2
    )
    
    # Generadores de datos mejorados
    train_gen, val_gen, test_gen, steps_per_epoch, val_steps, test_steps, data_splits = trainer.create_advanced_generators(
        balanced_df, image_paths, batch_size=BATCH_SIZE
    )
    
    # Crear un modelo mejorado
    model = trainer.create_improved_model()
    
    # Focal Loss ile derle
    trainer.compile_model_with_focal_loss(learning_rate=0.001)
    
    # Entrenamiento de modelos
    history = trainer.train_improved_model(
        train_gen, val_gen, steps_per_epoch, val_steps,
        epochs=EPOCHS,
        model_save_path='models/improved_balanced_7class_model.h5'
    )
    
    # Evaluación del modelo
    evaluation_results = trainer.evaluate_improved_model(test_gen, test_steps, data_splits)
    
    # Visualizaciones
    trainer.plot_improved_results(evaluation_results)
    
    # Conversión de TFLite
    tflite_path = trainer.convert_to_tflite(
        'models/improved_balanced_7class_model.h5',
        'models/flutter_assets/improved_balanced_7class_model.tflite'
    )
    
    print(f"\n🎉 ¡Modelo mejorado de 7 clases completado!")
    print("=" * 60)
    print("🔧 Mejoras:")
    print("  ✅ Pérdida focal (desequilibrio de datos resuelto)")
    print("  ✅ Muestreo equilibrado (todas las clases iguales)")
    print("  ✅ Aumento agresivo")
    print("  ✅ Arquitectura mejorada")
    print(f"📱 TFLite: {tflite_path}")
    print(f"🎯 Test Accuracy: {evaluation_results['test_results'][1]:.4f}")
    
    # Comparación final
    print(f"\n📊 Mejoras esperadas:")
    print(f"  🔸 Precisión previa: ~70% → Objetivo: >85%")
    print(f"  🔸 Clases débiles: akiec, bcc, df, vasc → Mucho mejor")
    print(f"  🔸 Rendimiento equilibrado en todas las clases")

if __name__ == "__main__":
    main()
