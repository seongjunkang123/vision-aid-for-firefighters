import pickle
import matplotlib.pyplot as plt
import information
import os
import glob
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

# Get all pickle files from saved_histories directory
history_files = glob.glob("saved_histories/dehazer_history_trial*.pkl")

def calculate_psnr_for_trial(trial_num):
    """Calculate PSNR values for a specific trial's output images"""
    psnr_values = []
    # Adjust these paths according to your directory structure
    dehazed_dir = f"results/trial_{trial_num}/dehazed"
    ground_truth_dir = "dataset/ground_truth"  # Adjust this path
    
    # Get sorted list of image files
    dehazed_files = sorted(glob.glob(os.path.join(dehazed_dir, "*.png")))
    ground_truth_files = sorted(glob.glob(os.path.join(ground_truth_dir, "*.png")))
    
    for dehazed_path, gt_path in zip(dehazed_files, ground_truth_files):
        # Load images
        dehazed = plt.imread(dehazed_path)
        ground_truth = plt.imread(gt_path)
        
        # Calculate PSNR
        psnr_value = psnr(ground_truth, dehazed, data_range=1.0)
        psnr_values.append(psnr_value)
    
    return np.mean(psnr_values)  # Return average PSNR for this trial

# Calculate and store PSNR values for each trial
psnr_results = {}
for file_path in history_files:
    trial_part = file_path.split('trial')[-1].split('.')[0]
    trial_num = int(trial_part.replace('_', ''))
    psnr_value = calculate_psnr_for_trial(trial_num)
    psnr_results[trial_num] = psnr_value
    print(f"Trial {trial_num} Average PSNR: {psnr_value:.2f} dB")

# Now you can plot both MAE and PSNR
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Plot MAE as before
for file_path in history_files:
    trial_part = file_path.split('trial')[-1].split('.')[0]
    trial_num = int(trial_part.replace('_', ''))
    with open(file_path, 'rb') as f:
        history = pickle.load(f)
    ax1.plot(history['mae'], label=f'MAE (Trial {trial_num})')

# Plot PSNR values as horizontal lines
for trial_num, psnr_value in psnr_results.items():
    ax2.axhline(y=psnr_value, label=f'PSNR (Trial {trial_num})')

# Configure plots
ax1.set_title("Mean Absolute Error Across All Trials")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MAE")
ax1.legend()
ax1.grid(True)

ax2.set_title("PSNR Values Across All Trials")
ax2.set_xlabel("Trial")
ax2.set_ylabel("PSNR (dB)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Print maximum validation accuracy for each trial
print("\nMaximum validation accuracies:")
for file_path in history_files:
    trial_part = file_path.split('trial')[-1].split('.')[0]
    trial_num = int(trial_part.replace('_', ''))
    with open(file_path, 'rb') as f:
        history = pickle.load(f)

