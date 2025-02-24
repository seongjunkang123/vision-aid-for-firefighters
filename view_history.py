import pickle
import matplotlib.pyplot as plt
import information

TRIAL_NUMBER = information.TRIAL_NUMBER
path = information.HISTORY_SAVE_PATH

with open(path, 'rb') as f:
    history = pickle.load(f)

plt.figure(figsize=(10,6))
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title("training and validation accuracy")
plt.xlabel("epoch")
plt.ylabel('accuracy')
plt.legend()
plt.grid(True)
plt.show()