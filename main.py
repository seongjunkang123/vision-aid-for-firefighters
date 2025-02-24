import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import pickle
from PIL import Image
import information

TRIAL_NUMBER = information.TRIALNUMBER
MODEL_SAVE_PATH = information.MODEL_SAVE_PATH
HISTORY_SAVE_PATH = information.HISTORY_SAVE_PATH
EPOCH = information.EPOCH
BATCH = information.BATCH

# get data from /BIPEDv2
dataset_path = 'BIPEDv2/BIPED/edges'

train_image_path = dataset_path + '/imgs/train/rgbr/real/'
train_label_path = dataset_path + '/edge_maps/train/rgbr/real/'

test_image_path = dataset_path +  '/imgs/test/rgbr/'
test_label_path = dataset_path +  '/edge_maps/test/rgbr/'

def load_dataset(directory, image_size=(256, 256), printName=False):
    print(directory + ': ')
    image_files = sorted([f for f in os.listdir(directory) if f.endswith('.png') or 
                                                        f.endswith('.jpg') or 
                                                        f.endswith('.jpeg')])
    length = len(image_files)
    images = np.empty((length, image_size[0], image_size[1], 1 if 'edge_maps'  in directory else 3), dtype='float32')

    for i, filename in enumerate(image_files):
        if printName: print(filename)
        image_path = os.path.join(directory, filename)
        image = Image.open(image_path)
        if 'edge_maps' in directory:
            image.convert('L')
        image = image.resize(image_size)

        if 'edge_maps' in directory:
            images[i, :, :, 0] = np.array(image) / 255.0
        else:
            images[i] = np.array(image) / 255.0

    return images

train_image = load_dataset(train_image_path)
train_label = load_dataset(train_label_path)
test_image = load_dataset(test_image_path)
test_label = load_dataset(test_label_path)

# preprocess data
train_image = train_image.astype('float32') / 255.0
train_label = train_label.astype('float32') / 255.0

test_image = test_image.astype('float32') / 255.0
test_label = test_label.astype('float32') / 255.0

train_image = np.expand_dims(train_image, axis=-1)
train_label = np.expand_dims(train_label, axis=-1)

test_image = np.expand_dims(test_image, axis=-1)
test_label = np.expand_dims(test_label, axis=-1)

# train_dataset = tf.data.Dataset.from_tensor_slices((train_image, train_label))
# train_dataset = train_dataset.shuffle(buffer_size=1024)

# test_dataset = tf.data.Dataset.from_tensor_slices((test_image, test_label))
# test_dataset = test_dataset.shuffle(buffer_size=1024)

# initialize model
def cnn_model(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    # Encoder (Downsampling)
    def encoder_block(x, filters, kernel_size=(3, 3), padding='same', activation='relu'):
        x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        return x

    # Decoder (Upsampling)
    def decoder_block(x, skip_features, filters, kernel_size=(3, 3), padding='same', activation='relu'):
        x = layers.Conv2DTranspose(filters, kernel_size, strides=(2, 2), padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.concatenate([x, skip_features])  # Skip connection
        x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        return x

    # Encoder Path
    e1 = encoder_block(inputs, 32)  # Reduced filters
    p1 = layers.MaxPooling2D((2, 2))(e1)

    e2 = encoder_block(p1, 64)  # Reduced filters
    p2 = layers.MaxPooling2D((2, 2))(e2)

    e3 = encoder_block(p2, 128)  # Reduced filters
    p3 = layers.MaxPooling2D((2, 2))(e3)

    # Bottleneck
    bottleneck = encoder_block(p3, 256)  # Reduced filters

    # Decoder Path
    d1 = decoder_block(bottleneck, e3, 128)  # Reduced filters
    d2 = decoder_block(d1, e2, 64)  # Reduced filters
    d3 = decoder_block(d2, e1, 32)  # Reduced filters

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(d3)

    # Build the model
    model = models.Model(inputs, outputs, name='EdgeDetectionModel')
    # print(model.summary())
    return model

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

# model compile and fit
model = cnn_model(input_shape=(256, 256, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(
    train_image, 
    train_label, 
    epochs=EPOCH,
    batch_size=BATCH_SIZE, 
    validation_split = 0.2,
    validation_batch_size=BATCH_SIZE, 
    callbacks=[early_stopping_callback, checkpoint_callback]
    )

# save model and history
model.save(MODEL_SAVE_PATH)

import pickle
with open(HISTORY_SAVE_PATH, 'wb') as file:
  pickle.dump(history.history, file)

test_loss, test_accuracy = model.evaluate(test_image, test_label)
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