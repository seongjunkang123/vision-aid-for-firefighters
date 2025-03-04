# imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import keras

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

from PIL import Image

import information
MODEL_SAVE_PATH = information.MODEL_SAVE_PATH
HISTORY_SAVE_PATH = information.HISTORY_SAVE_PATH
NUM_EPOCHS = information.NUM_EPOCHS
BATCH_SIZE = information.BATCH_SIZE
INPUT_SHAPE = information.INPUT_SHAPE

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

def load_trans_dataset(directory, image_size=(256, 256)):
    print(directory + ': ')
    image_files = sorted([f for f in os.listdir(directory) if f.endswith('.png') or
                                                        f.endswith('.jpg') or
                                                        f.endswith('.jpeg')])
    length = 2000
    images = np.empty((length, image_size[0], image_size[1], 1), dtype='float32')

    i = 0
    while i < length:
        filename = image_files[i * 10]
        image_path = os.path.join(directory, filename)
        image = Image.open(image_path)
        image = image.resize(image_size)
        image.convert('L')
        images[i, :, :, 0] = np.array(image) / 255.0

        i += 1

    return images

def load_clear_dataset(directory, image_size=(256, 256)):
    print(directory + ': ')
    image_files = sorted([f for f in os.listdir(directory) if f.endswith('.png') or
                                                        f.endswith('.jpg') or
                                                        f.endswith('.jpeg')])
    print(len(image_files))
    images = []

    i = 0
    while i < 2000:
        filename = image_files[i]
        image_path = os.path.join(directory, filename)
        image = Image.open(image_path)
        image = image.resize(image_size)
        images.append(np.array(image) / 255.0)

        i += 1

    return np.array(images)

# hazy_folder_path = 'RESIDE/hazy/'
# hazy_images = load_hazy_dataset(hazy_folder_path)
# print(hazy_images.shape)

trans_folder_path = 'RESIDE/trans/'
trans_images = load_trans_dataset(trans_folder_path)
print(trans_images.shape)

clear_folder_path = 'RESIDE/clear/'
clear_images = load_clear_dataset(clear_folder_path)
print(clear_images.shape)

x_train, x_test, y_train, y_test = train_test_split(trans_images, clear_images, test_size=0.2, random_state=42, shuffle=True)

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

def build_dehazing_model2(input_shape=(256, 256, 1)):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Encoder
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Decoder
    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Output layer (3 channels for RGB)
    outputs = layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs, name="Dehazer")
    return model

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model = build_dehazing_model2()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(  x_train,
                      y_train,
                      epochs=NUM_EPOCHS,
                      batch_size=BATCH_SIZE,
                      validation_split=0.2,
                      validation_batch_size=BATCH_SIZE,
                      callbacks=[early_stopping_callback])

model.save(MODEL_SAVE_PATH)

import pickle
with open(HISTORY_SAVE_PATH, 'wb') as file:
  pickle.dump(history.history, file)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
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