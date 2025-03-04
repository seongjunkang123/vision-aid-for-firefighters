import pickle
import matplotlib.pyplot as plt
import information

# Set your desired trial number
TRIAL_NUMBER = 3  # Change this to the specific trial number you want

# Construct the path dynamically
path = f"saved_histories/training_history_trial_{TRIAL_NUMBER}.pkl"

print(f"Using trial number: {TRIAL_NUMBER}")
print(f"Loading history from: {path}")

# Load history from the specified trial file
with open(path, 'rb') as f:
    history = pickle.load(f)

# Plot the training and validation accuracy
plt.figure(figsize=(10,6))
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title("Training and Validation Accuracy: Trial " + str(TRIAL_NUMBER))
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
