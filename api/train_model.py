import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model
import joblib

# ---------------------------
# 🔹 Load & Validate Data
# ---------------------------
try:
    data = pd.read_csv("/Users/alexandrakhreiche/Desktop/code/planwise_chatbots/recommendation-system/api/new_dataset.csv")  # Ensure this file is the cleaned dataset
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: File not found. Ensure 'new_dataset.csv' exists in the correct path.")
    exit()

# 🔹 Display dataset info
print("\n📊 Dataset Info:")
print(data.info())
print("\n🔍 First 5 Rows:")
print(data.head())

# ---------------------------
# 🔹 Data Preprocessing
# ---------------------------
# Drop the 'user_id' column since it's not needed for training
if 'user_id' in data.columns:
    data = data.drop(columns=['user_id'])

# Convert all columns to numeric values
data = data.apply(pd.to_numeric, errors='coerce')

# Drop any empty rows after conversion
data = data.dropna()

# Final dataset check
if data.shape[0] == 0:
    print("❌ Error: No data left after cleaning. Check CSV formatting.")
    exit()

print(f"✅ Data successfully cleaned! Shape: {data.shape}")

# ---------------------------
# 🔹 Feature Scaling (MinMax Scaling)
# ---------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(data.values)

# ---------------------------
# 🔹 Train-Test Split
# ---------------------------
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)
input_dim = X_train.shape[1]  # Number of categories

print(f"📊 Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# ---------------------------
# 🔹 Build the Autoencoder Model
# ---------------------------
inp = Input(shape=(input_dim,))
encoded = Dense(int(input_dim / 2), activation='relu')(inp)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(inp, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# ---------------------------
# 🔹 Train the Model
# ---------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

print("\n🚀 Training the Autoencoder...")
autoencoder.fit(
    X_train, X_train,
    epochs=5,
    batch_size=256,
    shuffle=True,
    validation_data=(X_test, X_test),
    callbacks=[early_stop]
)

# ---------------------------
# 🔹 Save Model & Scaler
# ---------------------------
autoencoder.save("autoencoder.h5")
joblib.dump(scaler, "scaler.save")

print("\n✅ Model and scaler saved successfully!")

# Define the custom loss function explicitly
custom_objects = {"mse": MeanSquaredError()}

# Load the trained model and scaler
autoencoder = load_model("autoencoder.h5", custom_objects=custom_objects)
scaler = joblib.load("scaler.save")

print("✅ Model and scaler loaded successfully!")


# Load new user data (replace this with real user data)
new_user_data = pd.DataFrame([{
    "resorts": 4.0,
    "burger/pizza shops": 3.5,
    "hotels/other lodgings": 4.5,
    "juice bars": 2.0,
    "beauty & spas": 3.0,
    "gardens": 4.2,
    "Amusement Parks": 1.5,
    "Farmer market": 4.1,
    "Market": 3.0,
    "Music halls": 2.5,
    "Nature": 4.8,
    "Tourist attractions": 4.6,
    "beaches": 4.0,
    "parks": 3.8,
    "theatres": 3.2,
    "museums": 3.9,
    "malls": 4.2,
    "restaurants": 4.5,
    "pubs/bars": 3.0,
    "local services": 2.9,
    "art galleries": 3.7,
    "dance clubs": 3.1,
    "swimming pools": 4.3,
    "bakeries": 3.0,
    "cafes": 3.2,
    "view points": 4.4,
    "monuments": 4.1,
    "zoo": 3.8,
    "supermarket": 4.5
}])

# Scale the data using the same scaler
new_user_scaled = scaler.transform(new_user_data)

print("✅ New user data prepared for prediction!")


# Pass the scaled data through the autoencoder
predicted_preferences = autoencoder.predict(new_user_scaled)

# Convert back to original scale
predicted_preferences_original = scaler.inverse_transform(predicted_preferences)

# Convert results into a DataFrame
predicted_df = pd.DataFrame(predicted_preferences_original, columns=new_user_data.columns)

print("\n🔮 **Predicted User Preferences:**")
print(predicted_df)

