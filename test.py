import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import cv2 as cv
from PIL import Image
import information
import random

model_path = information.MODEL_SAVE_PATH
print(model_path)
model = load_model(model_path)

test_dir = 'prediction_images/'

# Preprocess prediction image
images_files = sorted([f for f in os.listdir(test_dir) 
                       if f.endswith('.png') or f.endswith('jpg') or f.endswith('jpeg')])
index = random.randrange(0, len(images_files))
filename = images_files[index]
image_path = os.path.join(test_dir, filename)

image_size = (256, 256)
image = np.empty((image_size[0], image_size[1], 3), dtype='float32')

img = Image.open(image_path)
img = img.resize(image_size)
image = np.array(img) / 255.0
image = np.expand_dims(image, axis=0)
print(image.shape)

prediction = model.predict(image)

plt.figure(figsize=(10, 5))
# Plot input image
plt.subplot(1, 2, 1)
plt.imshow(np.squeeze(image, axis=0))
plt.title('Input Image')
plt.axis('off')

# Plot predicted edges
plt.subplot(1, 2, 2)
plt.imshow(np.squeeze(prediction, axis=0), cmap='gray')
plt.title('Predicted Edges')
plt.axis('off')

plt.show()