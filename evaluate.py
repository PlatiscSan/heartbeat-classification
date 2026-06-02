"""
Heartbeat Signal Classification - Evaluation Script
Evaluates the trained model on test set and generates classification metrics.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
							recall_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== Configuration ====================
DATA_PATH = "output/preprocessed_heartbeat.npz"
MODEL_PATH = "output/best_model.pth"
CONF_MATRIX_PATH = "output/confusion_matrix.png"
REPORT_PATH = "output/evaluation_report.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==================== Model Definition (must match train.py) ====================
class HeartbeatCNN(nn.Module):
	def __init__(self, num_classes=4):
		super(HeartbeatCNN, self).__init__()
		self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
		self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
		self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
		self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
		self.dropout1 = nn.Dropout(0.5)
		self.dropout2 = nn.Dropout(0.3)
		self.flatten_dim = 128 * 25   # 205 // 2 // 2 // 2 = 25
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

# ==================== Evaluation Function ====================
def evaluate(model, loader, device):
	model.eval()
	all_preds = []
	all_labels = []

	with torch.no_grad():
		for inputs, labels in loader:
			inputs, labels = inputs.to(device), labels.to(device)
			outputs = model(inputs)
			preds = torch.argmax(outputs, dim=1)
			all_preds.extend(preds.cpu().numpy())
			all_labels.extend(labels.cpu().numpy())

	return np.array(all_labels), np.array(all_preds)

# ==================== Main ====================
def main():
	# Load data
	print("Loading preprocessed data...")
	data = np.load(DATA_PATH)
	X_test = data['X_test']
	y_test = data['y_test']
	print(f"Test set shape: {X_test.shape}")

	# Create DataLoader
	from torch.utils.data import TensorDataset, DataLoader
	test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
								 torch.tensor(y_test, dtype=torch.long))
	test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

	# Load model
	model = HeartbeatCNN(num_classes=4).to(DEVICE)
	model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
	print(f"Model loaded from {MODEL_PATH}")

	# Evaluate
	y_true, y_pred = evaluate(model, test_loader, DEVICE)

	# Metrics
	acc = accuracy_score(y_true, y_pred)
	macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
	macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
	weighted_f1 = f1_score(y_true, y_pred, average='weighted')
	macro_f1 = f1_score(y_true, y_pred, average='macro')

	print("\n========== Evaluation Results ==========")
	print(f"Accuracy:			  {acc:.4f}")
	print(f"Macro Precision:	   {macro_prec:.4f}")
	print(f"Macro Recall:		  {macro_rec:.4f}")
	print(f"Macro F1-score:		{macro_f1:.4f}")
	print(f"Weighted F1-score:	 {weighted_f1:.4f}")
	print("========================================\n")

	# Per-class metrics
	class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
	print("Classification Report:")
	print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

	# Confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	plt.figure(figsize=(8, 6))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
				xticklabels=class_names, yticklabels=class_names)
	plt.xlabel('Predicted Label')
	plt.ylabel('True Label')
	plt.title('Confusion Matrix on Test Set')
	plt.tight_layout()
	plt.savefig(CONF_MATRIX_PATH, dpi=150)
	plt.close()
	print(f"Confusion matrix saved to {CONF_MATRIX_PATH}")

	# Save evaluation report
	with open(REPORT_PATH, 'w') as f:
		f.write("========== Evaluation Report ==========\n")
		f.write(f"Accuracy:			  {acc:.4f}\n")
		f.write(f"Macro Precision:	   {macro_prec:.4f}\n")
		f.write(f"Macro Recall:		  {macro_rec:.4f}\n")
		f.write(f"Macro F1-score:		{macro_f1:.4f}\n")
		f.write(f"Weighted F1-score:	 {weighted_f1:.4f}\n\n")
		f.write("Classification Report:\n")
		f.write(classification_report(y_true, y_pred, target_names=class_names, digits=4))
		f.write("\nConfusion matrix:\n")
		f.write(np.array2string(cm))
	print(f"Evaluation report saved to {REPORT_PATH}")

if __name__ == "__main__":
	main()