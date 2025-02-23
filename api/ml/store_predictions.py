import psycopg2
import pandas as pd
from .predict import get_predictions

def store_user_predictions(user_id, preferences):
    predictions = get_predictions(preferences)
    
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname="travel_db",
        user="postgres",
        password="alexa",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()
    
    # Store predictions in database
    # ... your existing storage logic ...
    
    cursor.close()
    conn.close()
    
    return predictions 