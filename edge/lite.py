import tensorflow as tf
import information

path = information.MODEL_SAVE_PATH
model = tf.keras.models.load_model(path)
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

