import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

print("="*60)
print("Testing Supabase PostgreSQL Connection")
print("="*60)

# Configuration from .env (Direct Connection)
config = {
    'dbname': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST', 'db.cpjmodytpeuybpcayzwk.supabase.co'),
    'port': os.getenv('DB_PORT', '5432'),
}

print(f"\nConnection Details:")
print(f"  Host: {config['host']}")
print(f"  Port: {config['port']}")
print(f"  Database: {config['dbname']}")
print(f"  User: {config['user']}")
print(f"  Password: {'*' * len(config['password']) if config['password'] else 'NOT SET'}")
print()

try:
    print("Attempting to connect...")
    conn = psycopg2.connect(
        dbname=config['dbname'],
        user=config['user'],
        password=config['password'],
        host=config['host'],
        port=config['port'],
        connect_timeout=10,
        sslmode='require'
    )
    print("✅ CONNECTION SUCCESSFUL!\n")
    
    cursor = conn.cursor()
    
    # Get PostgreSQL version
    cursor.execute("SELECT version();")
    version = cursor.fetchone()
    print(f"PostgreSQL Version:")
    print(f"  {version[0][:80]}...")
    print()
    
    # Get current database
    cursor.execute("SELECT current_database();")
    db = cursor.fetchone()
    print(f"Current Database: {db[0]}")
    
    # Get current user
    cursor.execute("SELECT current_user;")
    user = cursor.fetchone()
    print(f"Current User: {user[0]}")
    
    # List existing tables
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    tables = cursor.fetchall()
    print(f"\nExisting Tables ({len(tables)}):")
    if tables:
        for table in tables:
            print(f"  - {table[0]}")
    else:
        print("  (No tables yet)")
    
    cursor.close()
    conn.close()
    
    print(f"\n{'='*60}")
    print("✅ Database connection is working correctly!")
    print("You can now run: python manage.py migrate")
    print(f"{'='*60}\n")
    
except psycopg2.OperationalError as e:
    print(f"❌ CONNECTION FAILED!")
    print(f"\nError: {str(e)}")
    print("\n⚠️  TROUBLESHOOTING:")
    print("1. Verify your database password in Supabase Dashboard")
    print("2. Check if the host is reachable (firewall/network)")
    print("3. Ensure SSL is enabled in your Supabase project")
    print(f"\n{'='*60}\n")
    
except Exception as e:
    print(f"❌ UNEXPECTED ERROR: {type(e).__name__}")
    print(f"   {str(e)}")
    print(f"\n{'='*60}\n")
