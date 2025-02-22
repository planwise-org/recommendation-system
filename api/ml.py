import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# ✅ Fix: Ensure the loss function is known to Keras
custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}

# Load the model
model = load_model("autoencoder.h5", custom_objects=custom_objects)
scaler = joblib.load("scaler.save")

# Generate dummy input (Replace with real user data later)
input_data = np.random.uniform(1, 5, size=(1, 29))  # Simulated input

# Normalize input using scaler
input_scaled = scaler.transform(input_data)

# Get predictions
predicted_scaled = model.predict(input_scaled)

# Convert back to original scale
predicted_preferences = scaler.inverse_transform(predicted_scaled)

# Convert to DataFrame
columns = [
    "resorts", "burger/pizza shops", "hotels/other lodgings", "juice bars", "beauty & spas", 
    "gardens", "Amusement Parks", "Farmer market", "Market", "Music halls", "Nature", 
    "Tourist attractions", "beaches", "parks", "theatres", "museums", "malls", 
    "restaurants", "pubs/bars", "local services", "art galleries", "dance clubs", 
    "swimming pools", "bakeries", "cafes", "view points", "monuments", "zoo", "supermarket"
]

predicted_df = pd.DataFrame(predicted_preferences, columns=columns)

# Save predictions
predicted_df.to_csv("predicted_preferences.csv", index=False)
print("✅ Predicted preferences saved to 'predicted_preferences.csv'")
