import os
import psycopg2
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

print("🔍 Probando conexión a PostgreSQL de Supabase...\n")

# Configuración de conexión
db_config = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'connect_timeout': 10
}

print(f"Host: {db_config['host']}")
print(f"Port: {db_config['port']}")
print(f"User: {db_config['user']}")
print(f"Database: {db_config['dbname']}\n")

try:
    print("Intentando conectar...")
    conn = psycopg2.connect(**db_config)
    print("✅ Conexión exitosa!")
    
    # Probar una consulta simple
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    db_version = cursor.fetchone()
    print(f"\n📊 Versión de PostgreSQL: {db_version[0]}")
    
    # Verificar si existen tablas
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    tables = cursor.fetchall()
    
    if tables:
        print(f"\n📋 Tablas existentes ({len(tables)}):")
        for table in tables:
            print(f"  - {table[0]}")
    else:
        print("\n📋 No hay tablas en la base de datos (está vacía)")
    
    cursor.close()
    conn.close()
    print("\n✅ Conexión cerrada correctamente")
    
except psycopg2.OperationalError as e:
    print(f"❌ Error de conexión: {e}")
    print("\n💡 Posibles soluciones:")
    print("1. Verifica que la contraseña sea correcta en Supabase Dashboard")
    print("2. Asegúrate de que la IP esté permitida (desactiva 'IP restrictions' temporalmente)")
    print("3. Verifica que el proyecto de Supabase esté activo")
    
except Exception as e:
    print(f"❌ Error inesperado: {e}")
