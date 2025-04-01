import cv2
import numpy as np
import tensorflow as tf
import time

# Load the optimized UINT8 TFLite model
interpreter = tf.lite.Interpreter(model_path="quantized_model.tflite")
interpreter.allocate_tensors()

# Get input & output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Check if UINT8 model is being used
input_dtype = input_details[0]['dtype']
assert input_dtype == np.uint8, f"Expected UINT8 input, but got {input_dtype}"

# Get input shape
input_shape = input_details[0]['shape']
height, width = input_shape[1], input_shape[2]  # Model's expected input size

# Start video capture
cap = cv2.VideoCapture(0)  # 0 = Default webcam

while True:
    start_time = time.time()  # Measure FPS

    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR (OpenCV default) to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, (width, height))  # Resize to model input size

    # Convert to UINT8 (no need to normalize)
    input_data = resized_frame.astype(np.uint8)
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Convert output to 8-bit grayscale image
    output_data = np.squeeze(output_data)  # Remove batch dimension
    output_data = cv2.resize(output_data, (640, 480))  # Resize for display

    # Show output
    cv2.imshow("Edge Detection", output_data)

    # Calculate FPS
    fps = 1.0 / (time.time() - start_time)
    print(f"FPS: {fps:.2f}")

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
