import tensorflow as tf
import joblib
import numpy as np
from typing import Dict, Any

def load_model():
    custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}
    model = tf.keras.models.load_model("models/autoencoder.h5", custom_objects=custom_objects)
    scaler = joblib.load("models/scaler.save")
    return model, scaler

def get_predictions(user_preferences):
    model, scaler = load_model()
    # Convert preferences to input format
    input_data = np.array([[
        user_preferences.get('resorts', 3.0),
        user_preferences.get('burger/pizza shops', 3.0),
        # ... add all 29 features in the correct order
    ]])
    
    # Scale and predict
    input_scaled = scaler.transform(input_data)
    predicted_scaled = model.predict(input_scaled)
    predictions = scaler.inverse_transform(predicted_scaled)
    
    return predictions[0]  # Return first row of predictions

def get_predictions_simple(preferences: Dict[str, Any]) -> Dict[str, float]:
    """
    Simple recommendation function without using TensorFlow for now
    """
    # Default scores for different categories
    base_scores = {
        "restaurants": 4.0,
        "cafes": 3.5,
        "parks": 3.0,
        "museums": 3.0,
        "shopping": 2.5
    }
    
    # Adjust scores based on preferences
    if preferences.price_range:
        base_scores = {k: v * (5 - preferences.price_range) / 4 for k, v in base_scores.items()}
    
    if preferences.rating_minimum:
        base_scores = {k: v if v >= preferences.rating_minimum else 0 for k, v in base_scores.items()}
    
    return base_scores 