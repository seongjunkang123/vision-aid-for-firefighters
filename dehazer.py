import numpy as np
import cv2
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import os

from google.colab import drive
drive.mount('/content/drive')

from PIL import Image
def load_hazy_dataset(directory, image_size=(256, 256)):
    print(directory + ': ')
    image_files = sorted([f for f in os.listdir(directory) if f.endswith('.png') or
                                                        f.endswith('.jpg') or
                                                        f.endswith('.jpeg')])
    length = len(image_files)
    images = np.empty((length, image_size[0], image_size[1], 3), dtype='float32')

    for i, filename in enumerate(image_files):
        image_path = os.path.join(directory, filename)
        image = Image.open(image_path)
        image = image.resize(image_size)
        images[i] = np.array(image) / 255.0
        print("Loaded image ", i)

    return images

def load_clear_dataset(directory, image_size=(256, 256)):
    print(directory + ': ')
    image_files = sorted([f for f in os.listdir(directory) if f.endswith('.png') or
                                                        f.endswith('.jpg') or
                                                        f.endswith('.jpeg')])
    print(len(image_files))
    images = []

    for i, filename in enumerate(image_files):
        image_path = os.path.join(directory, filename)
        image = Image.open(image_path)
        image = image.resize(image_size)
        for j in range(10):
            images.append(np.array(image) / 255.0)
        print("Loaded image ", i)

    return np.array(images)

hazy_folder_path = '/content/drive/Othercomputers/My MacBook Pro/Reside/hazy'
hazy_images = load_hazy_dataset(hazy_folder_path)
print(hazy_images.shape)
print(hazy_images[0])

clear_folder_path = '/content/drive/Othercomputers/My MacBook Pro/Reside/clear'
clear_images = load_clear_dataset(clear_folder_path)
print(clear_images.shape)
print(clear_images[0])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(hazy_images, clear_images, test_size=0.2, random_state=42, shuffle=True)

NUM_EPOCHS = 20
BATCH_SIZE = 32
INPUT_SHAPE = (256, 256, 3)
TRIAL = 0   # MODIFY THIS BEFORE EVERY RUN
MODEL_SAVE_PATH = f'/content/drive/MyDrive/Synopsys 2024-2025 SIV/Firefighters/Saved Models/dehazer_trial_{TRIAL}.keras'
HISTORY_SAVE_PATH = f'/content/drive/MyDrive/Synopsys 2024-2025 SIV/Firefighters/Saved Models/dehazer_history_trial{TRIAL}.pkl'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
def build_dehazing_model():
    model = keras.Sequential([
        layers.Input(shape=INPUT_SHAPE),

        # Convolutional Layers (Feature Extraction)
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),

        # Convolutional Transpose Layers (Upsampling)
        layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),

        # Output Layer (Clear Image)
        layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same') #sigmoid outputs values between 0 and 1.
    ])
    return model

model = build_dehazing_model()
model.compile(optimizer='adam', loss='binary_crossentropy')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint_callback = ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train,
                      y_train,
                      epochs=NUM_EPOCHS,
                      batch_size=BATCH_SIZE,
                      validation_split=0.2,
                      validation_batch_size=BATCH_SIZE,
                      callbacks=[early_stopping_callback, checkpoint_callback])

model.save(MODEL_SAVE_PATH)
import pickle
with open(HISTORY_SAVE_PATH, 'wb') as file:
  pickle.dump(history.history, file)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Plotting the history
plt.figure(figsize=(10,6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("training and validation accuracy")
plt.xlabel("epoch")
plt.ylabel('accuracy')
plt.legend()
plt.grid(True)
plt.show()
