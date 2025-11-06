#!/usr/bin/env python3
"""
Script para probar la carga del modelo actualizado en Django
"""
import os
import sys
import django

# Configurar Django
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'skin_disease_project.settings')
django.setup()

# Suprimir warnings de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 70)
print("🧪 PRUEBA DE CARGA DEL MODELO ACTUALIZADO")
print("=" * 70)

try:
    from skin_detector.predictor import get_predictor
    
    print("\n1️⃣ Obteniendo instancia del predictor...")
    predictor = get_predictor()
    
    print("\n2️⃣ Verificando modelo cargado...")
    summary = predictor.get_model_summary()
    
    if 'error' in summary:
        print(f"❌ Error: {summary['error']}")
    else:
        print("✅ Modelo cargado exitosamente!")
        print(f"\n📊 Resumen del modelo:")
        print(f"   - Capas totales: {summary['total_layers']}")
        print(f"   - Input shape: {summary['input_shape']}")
        print(f"   - Output shape: {summary['output_shape']}")
        print(f"   - Parámetros: {summary['total_params']:,}")
        print(f"   - Clases: {summary['classes']}")
        
        print(f"\n🏷️ Nombres de clases:")
        for idx, name in summary['class_names'].items():
            print(f"   {idx}: {name}")
    
    print("\n" + "=" * 70)
    print("✅ PRUEBA COMPLETADA EXITOSAMENTE")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    import traceback
    print(f"\nTraceback completo:")
    print(traceback.format_exc())
    print("\n" + "=" * 70)
    print("❌ PRUEBA FALLIDA")
    print("=" * 70)
    sys.exit(1)
