"""
abs_eval.py - Competition Metric Evaluation
Computes the official Tianchi abs-sum metric: sum of absolute differences
between predicted probabilities and true one-hot labels.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os

# ==================== Configuration ====================
DATA_PATH = "output/preprocessed_heartbeat.npz"
MODEL_PATH = "output/best_model.pth"
OUTPUT_TXT = "output/abs_sum_result.txt"
BATCH_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==================== Model Definition (must match training) ====================
class HeartbeatCNN(nn.Module):
	def __init__(self, num_classes=4):
		super(HeartbeatCNN, self).__init__()
		self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
		self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
		self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
		self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
		self.dropout1 = nn.Dropout(0.5)
		self.dropout2 = nn.Dropout(0.3)
		# Input length 205 -> after 3 pools: 205//2//2//2 = 25
		self.flatten_dim = 128 * 25
		self.fc1 = nn.Linear(self.flatten_dim, 128)
		self.fc2 = nn.Linear(128, num_classes)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.pool(self.relu(self.conv1(x)))
		x = self.pool(self.relu(self.conv2(x)))
		x = self.pool(self.relu(self.conv3(x)))
		x = x.view(x.size(0), -1)
		x = self.dropout1(self.relu(self.fc1(x)))
		x = self.dropout2(x)
		x = self.fc2(x)
		return x

# ==================== Main Evaluation ====================
def main():
	# 1. Load data
	print("Loading preprocessed data...")
	if not os.path.exists(DATA_PATH):
		print(f"Error: {DATA_PATH} not found. Please run analyzation.py first.")
		return
	data = np.load(DATA_PATH)
	X_test = data['X_test']
	y_test = data['y_test'].astype(np.int64)
	print(f"Test set shape: {X_test.shape}, labels shape: {y_test.shape}")

	# 2. Create DataLoader
	test_dataset = TensorDataset(
		torch.tensor(X_test, dtype=torch.float32),
		torch.tensor(y_test, dtype=torch.long)
	)
	test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

	# 3. Load model
	print(f"Loading model from {MODEL_PATH}...")
	if not os.path.exists(MODEL_PATH):
		print(f"Error: {MODEL_PATH} not found. Please run train.py first.")
		return
	model = HeartbeatCNN(num_classes=4).to(DEVICE)
	model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
	model.eval()
	print("Model loaded successfully.")

	# 4. Get predicted probabilities
	all_probas = []
	all_labels = []
	with torch.no_grad():
		for inputs, labels in test_loader:
			inputs = inputs.to(DEVICE)
			outputs = model(inputs)
			probas = F.softmax(outputs, dim=1)  # convert logits to probabilities
			all_probas.append(probas.cpu().numpy())
			all_labels.append(labels.numpy())
	y_pred_proba = np.vstack(all_probas)
	y_true = np.concatenate(all_labels)
	print(f"Predicted probabilities shape: {y_pred_proba.shape}")

	# 5. Compute abs-sum
	n_samples = len(y_true)
	# Convert integer labels to one-hot
	y_onehot = np.zeros((n_samples, 4))
	y_onehot[np.arange(n_samples), y_true] = 1.0

	# Absolute differences per sample per class
	abs_diff = np.abs(y_pred_proba - y_onehot)
	sample_abs_sum = np.sum(abs_diff, axis=1)   # sum over 4 classes
	total_abs_sum = np.sum(sample_abs_sum)
	avg_abs_sum = total_abs_sum / n_samples

	# 6. Output results
	print("\n" + "="*50)
	print("COMPETITION METRIC: abs-sum")
	print("="*50)
	print(f"Number of test samples: {n_samples}")
	print(f"Total abs-sum: {total_abs_sum:.4f}")
	print(f"Average abs-sum per sample: {avg_abs_sum:.6f}")
	print("="*50)

	# Also show per-class average predicted probability (useful analysis)
	print("\nPer-class average predicted probability (on test set):")
	class_names = ['Normal (0)', 'Ventricular (1)', 'Supraventricular (2)', 'Fusion (3)']
	for c in range(4):
		mask = (y_true == c)
		if np.sum(mask) > 0:
			avg_proba_c = np.mean(y_pred_proba[mask, c])
			avg_abs_error_c = np.mean(abs_diff[mask, c])
			print(f"  {class_names[c]}: avg proba = {avg_proba_c:.4f}, avg abs error = {avg_abs_error_c:.4f}")
	
	# 7. Save results to file
	with open(OUTPUT_TXT, 'w') as f:
		f.write("abs-sum Evaluation Results\n")
		f.write("=========================\n")
		f.write(f"Model: {MODEL_PATH}\n")
		f.write(f"Test samples: {n_samples}\n")
		f.write(f"Total abs-sum: {total_abs_sum:.4f}\n")
		f.write(f"Average abs-sum per sample: {avg_abs_sum:.6f}\n\n")
		f.write("Per-class details:\n")
		for c in range(4):
			mask = (y_true == c)
			if np.sum(mask) > 0:
				avg_proba_c = np.mean(y_pred_proba[mask, c])
				avg_abs_error_c = np.mean(abs_diff[mask, c])
				f.write(f"  {class_names[c]}: samples={np.sum(mask)}, avg proba={avg_proba_c:.4f}, avg abs error={avg_abs_error_c:.4f}\n")
		f.write("\nFirst 10 sample predictions (proba and true):\n")
		for i in range(min(10, n_samples)):
			f.write(f"Sample {i}: true={y_true[i]}, pred_proba={y_pred_proba[i]}, abs_sum={sample_abs_sum[i]:.4f}\n")
	print(f"\nResults saved to {OUTPUT_TXT}")

if __name__ == "__main__":
	main()