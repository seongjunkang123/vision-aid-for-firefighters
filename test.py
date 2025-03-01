# removing tf warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# imports
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2 as cv
from PIL import Image
import information
import random

model_path = information.MODEL_SAVE_PATH
model = load_model(model_path)

test_dir = 'prediction_images/'

# Preprocess prediction image
images_files = sorted([f for f in os.listdir(test_dir) 
                       if f.endswith('.png') or f.endswith('jpg') or f.endswith('jpeg')])

image_size = (256, 256)
num_samples = len(images_files)
i = 0

plt.figure(figsize=(15, 5))

for filename in images_files:
    image_path = os.path.join(test_dir, filename)

    img = Image.open(image_path)
    img = img.resize(image_size)

    image = np.array(img) / 255.0
    image = np.expand_dims(image, axis=0)  # Shape becomes (1, 256, 256, 3)
    
    print("Image shape:", image.shape)

    prediction = model.predict(image)
    print("Prediction shape:", prediction.shape)
    
    # Remove batch dimension from image for visualization
    image = np.squeeze(image, axis=0)  # Shape becomes (256, 256, 3)
    
    # Plot the image
    plt.subplot(2, num_samples, i + 1)
    plt.imshow(image)
    plt.title(f"Image {i + 1}")
    plt.axis('off')

    # Plot the corresponding label
    plt.subplot(2, num_samples, i + 1 + num_samples)
    plt.imshow(np.squeeze(prediction), cmap='gray')  # Squeeze prediction if needed
    plt.title(f"Prediction {i + 1}")
    plt.axis('off')
    plt.tight_layout()

    i += 1

plt.show()