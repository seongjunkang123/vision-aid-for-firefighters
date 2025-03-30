import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="your_model.tflite")
interpreter.allocate_tensors()

# Get input & output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input shape
input_shape = input_details[0]['shape']
height, width = input_shape[1], input_shape[2]  # Model's expected input dimensions

# Start video capture
cap = cv2.VideoCapture(0)  # 0 = Default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR (OpenCV default) to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, (width, height))  # Resize to model's input size

    # Normalize input for the model (if required)
    input_data = resized_frame.astype(np.float32) / 255.0  # Scale to [0,1]
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

    # Set tensor and run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Post-process the output (normalize & convert to 8-bit image)
    output_data = np.squeeze(output_data)  # Remove batch dimension
    output_data = (output_data - output_data.min()) / (output_data.max() - output_data.min()) * 255
    output_data = output_data.astype(np.uint8)

    # Display the grayscale edge-detected output
    cv2.imshow("Edge Detection", output_data)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
