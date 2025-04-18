import os
from dotenv import load_dotenv
from supabase import create_client
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime
import bcrypt  # Use bcrypt directly instead of passlib

# Load environment variables
load_dotenv()

# Supabase connection
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Local PostgreSQL connection
LOCAL_DB = {
    'dbname': 'planwise',
    'user': 'user',
    'password': 'password',
    'host': 'localhost',
    'port': '5431'
}

def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt).decode()

def create_default_user():
    """Create a default user if none exists"""
    default_user = {
        "name": "Default User",
        "email": "default@example.com",
        "hash_password": hash_password("defaultpassword"),
        "preferences": json.dumps({"initialized": False})
    }
    
    try:
        print("Attempting to create default user:", default_user)
        result = supabase.table("user").insert(default_user).execute()
        print("Created default user")
        return result.data[0]["id"] if result.data else None
    except Exception as e:
        print(f"Error creating default user: {e}")
        return None

def clean_row_data(row):
    data = dict(row)
    # Convert datetime objects to ISO format
    for key, value in data.items():
        if isinstance(value, datetime):
            data[key] = value.isoformat()
    return data

def migrate_table(table_name, local_conn):
    print(f"\nMigrating {table_name}...")
    
    cursor = local_conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    
    if not rows:
        print(f"No data found in {table_name}")
        if table_name == "user":
            return create_default_user()
        return
    
    print(f"Found {len(rows)} rows in {table_name}")
    
    for row in rows:
        try:
            clean_data = clean_row_data(row)
            
            # If it's the user table and the data doesn't match our schema
            if table_name == "user" and (len(clean_data) == 1 or "name" not in clean_data):
                print("Invalid user data format, creating default user instead")
                return create_default_user()
            
            print(f"Attempting to insert data into {table_name}:", clean_data)
            result = supabase.table(table_name).insert(clean_data).execute()
            print(f"Inserted row in {table_name}: ID {clean_data.get('id', 'N/A')}")
            
            if table_name == "user":
                return result.data[0]["id"] if result.data else None
            
        except Exception as e:
            print(f"Error inserting row in {table_name}: {e}")
            print(f"Problematic data: {clean_data}")
            if table_name == "user":
                return create_default_user()

def main():
    try:
        # Connect to local database
        print("Connecting to local database...")
        local_conn = psycopg2.connect(**LOCAL_DB)
        
        # Migration order (respecting foreign key constraints)
        user_id = migrate_table("user", local_conn)
        if not user_id:
            print("Failed to create or migrate user. Stopping migration.")
            return
            
        # Continue with other tables
        for table in ['place', 'review', 'recommendation']:
            migrate_table(table, local_conn)
            
    except Exception as e:
        print(f"Migration error: {e}")
    finally:
        if 'local_conn' in locals():
            local_conn.close()

if __name__ == "__main__":
    main()