import tensorflow as tf
import information
import lite_info

path = information.MODEL_SAVE_PATH
model = tf.keras.models.load_model(path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter._experimental_lower_tensor_list_ops = True

tflite_model = converter.convert()

with open(lite_info.MODEL_SAVE_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"Model Version {lite_info.VERSION}")
lite_info.VERSION += 1