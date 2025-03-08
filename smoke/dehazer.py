# imports
import numpy as np
import cv2 as cv
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import os
import information
import random

# information calling
NUM_EPOCHS = information.NUM_EPOCHS
BATCH_SIZE = information.BATCH_SIZE
INPUT_SHAPE = information.INPUT_SHAPE
TRIAL = information.TRIAL
MODEL_SAVE_PATH = information.MODEL_SAVE_PATH
HISTORY_SAVE_PATH = information.HISTORY_SAVE_PATH

"""#Data"""

from PIL import Image
def load_hazy_dataset(directory, image_size=(256, 256)):
    print(directory + ': ')
    image_files = sorted([f for f in os.listdir(directory) if f.endswith('.png') or
                                                        f.endswith('.jpg') or
                                                        f.endswith('.jpeg')])
    images = []

    for i, filename in enumerate(image_files):
        image_path = os.path.join(directory, filename)
        image = Image.open(image_path)
        image = image.resize(image_size)
        images.append(np.array(image) / 255.0)
        # print("Loaded image ", i)

    return np.array(images)

from PIL import Image
def load_clear_dataset(directory, image_size=(256, 256)):
    print(directory + ': ')
    image_files = sorted([f for f in os.listdir(directory) if f.endswith('.png') or
                                                        f.endswith('.jpg') or
                                                        f.endswith('.jpeg')])
    images = []

    for i, filename in enumerate(image_files):
        image_path = os.path.join(directory, filename)
        image = Image.open(image_path)
        image = image.resize(image_size)
        images.append(np.array(image) / 255.0)
        # print("Loaded image ", i)

    return np.array(images)

hazy_folder_path = 'SMOKE/train/hazy/'
hazy_images = load_hazy_dataset(hazy_folder_path)

clear_folder_path = 'SMOKE/train/clean/'
clear_images = load_clear_dataset(clear_folder_path)

# data augmentation
def random_flip(image, label):
    if random.random() > 0.5:
        image = np.fliplr(image)
        label = np.fliplr(label)
    if random.random() > 0.5:
        image = np.flipud(image)
        label = np.flipud(label)
    return image, label

def random_rotate(image, label):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    angle = random.uniform(-180, 180)

    M = cv.getRotationMatrix2D(center, angle, 1.0)

    image = cv.warpAffine(image, M, (w, h), flags=cv.INTER_LINEAR)
    label = cv.warpAffine(label, M, (w, h), flags=cv.INTER_LINEAR)

    return image, label

def augment_image_and_label(image, label):
    image, label = random_flip(image, label)
    image, label = random_rotate(image, label)
    return image, label

def augment_dataset(images, labels):
    augmented_images = []
    augmented_labels = []

    multiple = 6
    for image, label in zip(images, labels):
        for i in range(multiple):            
            augmented_image, augmented_label = augment_image_and_label(image, label)
            augmented_images.append(augmented_image)
            augmented_labels.append(augmented_label)
            
    return np.array(augmented_images), np.array(augmented_labels)

augmented_train_image, augmented_train_label = augment_dataset(hazy_images, clear_images)
if augmented_train_label.ndim == 3:
    augmented_train_label = np.expand_dims(augmented_train_label, axis=-1)

print (augmented_train_image.shape, augmented_train_label.shape)

combined_train_image = np.concatenate([hazy_images, augmented_train_image], axis=0)
combined_train_label = np.concatenate([clear_images, augmented_train_label], axis=0)

# Considering the size of the dataset, data augmentation is probably unneccessary
# and would take a lot of computing resources

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(combined_train_image, combined_train_label, test_size=0.2, random_state=42, shuffle=True)

"""#Model Training"""
from tensorflow.keras import layers, models, regularizers

def build_dehazing_model(input_shape=INPUT_SHAPE):
    inputs = tf.keras.Input(shape=input_shape)

    #Encoder
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)  # Added dropout with 0.5 probability
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)  # Added dropout with 0.5 probability

    #Decoder
    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    #Output layer (3 channels for RGB)
    outputs = layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid')(x)

    model = models.Model(inputs, outputs, name="Dehazer")
    return model

model = build_dehazing_model()
model.summary()

# callbacks
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

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss=["mse"], metrics=["mae"])

history = model.fit(X_train,
                      y_train,
                      epochs=NUM_EPOCHS,
                      batch_size=BATCH_SIZE,
                      validation_split=0.2,
                      validation_batch_size=BATCH_SIZE)

model.save(MODEL_SAVE_PATH)

import pickle
with open(HISTORY_SAVE_PATH, 'wb') as file:
  pickle.dump(history.history, file)