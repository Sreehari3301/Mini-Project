import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def create_dummy_model():
    # Define a simple CNN for 224x224 RGB images
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(6, activation='softmax') # 6 classes as defined in app.py
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Save it as sign_model.keras
    model.save('sign_model.keras')
    print("Dummy model created and saved as 'sign_model.keras'")

if __name__ == "__main__":
    create_dummy_model()
