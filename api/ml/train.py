import tensorflow as tf
from .preprocessing import clean_dataset, prepare_for_training

def train_model(data_path):
    # Training logic here
    model = build_model()
    model.fit(...)
    model.save("models/autoencoder.h5") 