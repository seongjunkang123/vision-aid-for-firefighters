import tensorflow as tf
import information

path = information.MODEL_SAVE_PATH
model = tf.keras.models.load_model(path)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()
with open('model_1.tflite', 'wb') as f:
    f.write(tflite_model)

