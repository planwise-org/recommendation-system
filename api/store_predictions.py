import psycopg2
import pandas as pd
import joblib

# Load the predicted preferences
predicted_df = pd.read_csv("predicted_preferences.csv")  # Ensure you save the predictions

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="travel_db",
    user="postgres",
    password="alexa",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# Convert the predicted DataFrame into a dictionary
predicted_dict = predicted_df.iloc[0].to_dict()

# Insert the predicted values into user_preferences
query = """
INSERT INTO user_preferences (user_id, resorts, burger_pizza_shops, hotels_other_lodgings, juice_bars, beauty_spas, gardens, amusement_parks,
                              farmer_market, market, music_halls, nature, tourist_attractions, beaches, parks, theatres, museums, malls,
                              restaurants, pubs_bars, local_services, art_galleries, dance_clubs, swimming_pools, bakeries, cafes,
                              view_points, monuments, zoo, supermarket)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

values = ("new_user_1", *predicted_dict.values())  # Assign a user ID

cursor.execute(query, values)
conn.commit()
cursor.close()
conn.close()

print("✅ Predicted user preferences saved to database!")
