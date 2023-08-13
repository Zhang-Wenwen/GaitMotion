import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("./transfer_results/test.csv")

# Extract predictions and ground truth
predictions = data.iloc[:, 0].values
ground_truth = data.iloc[:, 1].values

# Calculate MAE and MSE
mae = np.mean(np.abs(predictions - ground_truth))
mse = np.mean((predictions - ground_truth)**2)

# Plot predictions vs. ground truth
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.scatter(ground_truth, predictions, alpha=0.6)
plt.plot([min(ground_truth), max(ground_truth)], [min(ground_truth), max(ground_truth)], color='red')  # diagonal line
plt.xlabel('Ground Truth')
plt.ylabel('Predictions')
plt.title('Predictions vs Ground Truth')

# Plot distribution of prediction errors
errors = predictions - ground_truth

plt.subplot(1, 2, 2)
plt.hist(errors, bins=50, color='blue', alpha=0.7)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')

plt.tight_layout()
plt.show()

