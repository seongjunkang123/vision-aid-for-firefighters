import tensorflow as tf
import information
import lite_info
import os
from PIL import Image
import numpy as np

path = information.MODEL_SAVE_PATH
model = tf.keras.models.load_model(path)

def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))  # Adjust size to match your model's input
    img = np.array(img) / 255.0  # Normalize to [0,1]
    return img[np.newaxis, ...]  # Add batch dimension to make shape (1, 256, 256, 3)

def representative_dataset():
    data_path = "BIPEDv2/BIPED/edges/imgs/train/rgbr/real"
    image_files = os.listdir(data_path)[:100]  # Use first 100 images
    for image_file in image_files:
        image_path = os.path.join(data_path, image_file)
        data = load_and_preprocess_image(image_path)
        yield [data.astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
# converter._experimental_lower_tensor_list_ops = True

tflite_model = converter.convert()

with open(lite_info.MODEL_SAVE_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"Model Version {lite_info.VERSION}")
lite_info.VERSION += 1