import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Initialize camera
cap = cv2.VideoCapture(0)  # Use 0 for default camera

# Load TensorFlow Lite model
model_path = 'your_model.tflite'  # Replace with your model path
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame for your model
    input_shape = input_details[0]['shape']
    input_data = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(input_data, axis=0)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Process output (customize based on your model)
    # Example: draw bounding boxes, display class labels, etc.
    
    # Display the resulting frame
    cv2.imshow('Live Inference', frame)
    
    # Break loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()