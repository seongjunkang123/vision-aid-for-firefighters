import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Initialize camera
cap = cv2.VideoCapture(0)

# Load TFLite model
interpreter = tflite.Interpreter(model_path='converted_model.tflite')
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame (adjust based on your model's requirements)
    input_shape = input_details[0]['shape'][1:3]  # e.g., (224, 224)
    input_data = cv2.resize(frame, input_shape)
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

    # Normalize if needed (e.g., [0, 255] -> [-1, 1])
    # input_data = (input_data / 127.5) - 1.0

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    # Display results
    cv2.putText(frame, f"Prediction: {np.argmax(predictions)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Live Inference', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()