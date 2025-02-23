import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # <--- For plotting
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# ---------------------------
# 1) Load and preprocess data
# ---------------------------
data = pd.read_csv("new_dataset.csv", sep=';', engine='python', skiprows=1)
data.columns = data.columns.str.strip().str.lower()

if 'user_id' in data.columns:
    data = data.drop(columns=['user_id'])

for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna()

X = data.values.astype('float32')
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# Dimensions
input_dim = X_train.shape[1]
encoding_dim = 16

# ---------------------------
# 2) Deeper autoencoder
# ---------------------------
def build_deeper_autoencoder(input_dim, encoding_dim=16):
    inp = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inp)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    encoded = Dense(encoding_dim, activation='relu')(x)

    y = Dense(32, activation='relu')(encoded)
    y = Dropout(0.2)(y)
    y = Dense(64, activation='relu')(y)
    decoded = Dense(input_dim, activation='sigmoid')(y)

    return Model(inp, decoded)

base_model = build_deeper_autoencoder(input_dim, encoding_dim)

# ---------------------------
# 3) Weighted Denoising Autoencoder
# ---------------------------
class WeightedDenoisingAutoencoder(Model):
    def __init__(self, base_model, missing_prob=0.5, weight_known=5.0, **kwargs):
        super(WeightedDenoisingAutoencoder, self).__init__(**kwargs)
        self.base_model = base_model
        self.missing_prob = missing_prob
        self.weight_known = weight_known
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        x, y = data  # x == y for an autoencoder
        # Random binary mask
        mask = tf.cast(tf.random.uniform(tf.shape(x)) > self.missing_prob, tf.float32)
        x_masked = mask * x

        with tf.GradientTape() as tape:
            y_pred = self.base_model(x_masked, training=True)

            # Weighted MSE
            known_diff = mask * (x - y_pred)
            missing_diff = (1 - mask) * (x - y_pred)

            known_mse = tf.reduce_mean(tf.square(known_diff)) * self.weight_known
            missing_mse = tf.reduce_mean(tf.square(missing_diff))
            loss = known_mse + missing_mse

        grads = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        mask = tf.cast(tf.random.uniform(tf.shape(x)) > self.missing_prob, tf.float32)
        x_masked = mask * x
        y_pred = self.base_model(x_masked, training=False)

        known_diff = mask * (x - y_pred)
        missing_diff = (1 - mask) * (x - y_pred)

        known_mse = tf.reduce_mean(tf.square(known_diff)) * self.weight_known
        missing_mse = tf.reduce_mean(tf.square(missing_diff))
        loss = known_mse + missing_mse

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


w_dae = WeightedDenoisingAutoencoder(base_model, missing_prob=0.5, weight_known=5.0)
w_dae.compile(optimizer='adam')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ---------------------------
# 4) Train the model and capture the History object
# ---------------------------
print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)

history = w_dae.fit(
    X_train,
    X_train,
    epochs=3, # 3 is the value for the minimum validation loss
    batch_size=256,
    shuffle=True,
    validation_data=(X_test, X_test),
    callbacks=[early_stop]
)

# ---------------------------
# 5) Visualize Training
# ---------------------------
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Weighted Denoising Autoencoder Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()

# ---------------------------
# 6) Save the underlying base_model + scaler
# ---------------------------
base_model.save("autoencoder.h5")
joblib.dump(scaler, "scaler.save")
print("Deeper Weighted Denoising Autoencoder and scaler saved!")
