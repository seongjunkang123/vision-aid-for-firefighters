import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your Keras model
model = load_model('your_model.h5')  # or .keras

# Initialize camera
cap = cv2.VideoCapture(0)  # For Pi Camera: cv2.VideoCapture(0, cv2.CAP_V4L2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame (adjust to match model's expected input)
    input_shape = model.input_shape[1:3]  # e.g., (224, 224)
    input_data = cv2.resize(frame, input_shape)
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

    # Normalize if needed (e.g., [0, 255] -> [0, 1] or [-1, 1])
    # input_data = input_data / 255.0  # For [0, 1] normalization
    # input_data = (input_data / 127.5) - 1.0  # For [-1, 1] normalization

    # Run inference
    predictions = model.predict(input_data)

    # Process predictions (example for classification)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]

    # Display results
    cv2.putText(frame, f"Class: {class_idx}, Conf: {confidence:.2f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Keras Model Live', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
