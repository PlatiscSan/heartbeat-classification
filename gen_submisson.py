"""
generate_submission.py
Generate submission CSV file in Tianchi competition format.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# ==================== Configuration ====================
ORIGINAL_DATA_PATH = "train.csv"
MODEL_PATH = "output/best_model.pth"
OUTPUT_CSV = "submission.csv"
BATCH_SIZE = 64
RANDOM_SEED = 42
TEST_SIZE = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==================== Parse signals (same as in analyzation.py) ====================
def parse_signals(signal_str):
	return np.array([float(x) for x in signal_str.split(',')])

def standardize(signal):
	mu = signal.mean()
	sigma = signal.std()
	if sigma < 1e-6:
		return signal - mu
	return (signal - mu) / sigma

# ==================== Model Definition ====================
class HeartbeatCNN(nn.Module):
	def __init__(self, num_classes=4):
		super(HeartbeatCNN, self).__init__()
		self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
		self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
		self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
		self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
		self.dropout1 = nn.Dropout(0.5)
		self.dropout2 = nn.Dropout(0.3)
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

# ==================== Main ====================
def main():
	# 1. Load original data (including id)
	print("Loading original data...")
	df = pd.read_csv(ORIGINAL_DATA_PATH)
	print(f"Original dataset size: {len(df)}")

	# 2. Parse signals and standardize
	print("Parsing and standardizing signals...")
	X = np.array([parse_signals(s) for s in df['heartbeat_signals']])
	X = np.array([standardize(sig) for sig in X])
	X = X[:, np.newaxis, :]   # add channel dimension -> (n, 1, 205)
	y = df['label'].values
	ids = df['id'].values

	# 3. Reproduce the same train/test split (stratified, 8:2)
	print(f"Splitting data (test_size={TEST_SIZE}, random_state={RANDOM_SEED})...")
	_, X_test, _, y_test, _, ids_test = train_test_split(
		X, y, ids, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
	)
	print(f"Test set size: {X_test.shape[0]}")

	# 4. Load model
	model = HeartbeatCNN(num_classes=4).to(DEVICE)
	model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
	model.eval()
	print(f"Model loaded from {MODEL_PATH}")

	# 5. Predict probabilities on test set
	test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))
	test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

	all_probas = []
	with torch.no_grad():
		for inputs in test_loader:
			inputs = inputs[0].to(DEVICE)
			outputs = model(inputs)
			probas = F.softmax(outputs, dim=1)
			all_probas.append(probas.cpu().numpy())
	y_pred_proba = np.vstack(all_probas)
	print(f"Predicted probabilities shape: {y_pred_proba.shape}")

	# 6. Create submission DataFrame
	submission = pd.DataFrame({
		'id': ids_test,
		'1abe1_0': y_pred_proba[:, 0],
		'1abe1_1': y_pred_proba[:, 1],
		'1abe1_2': y_pred_proba[:, 2],
		'1abe1_3': y_pred_proba[:, 3]
	})

	# 7. Save to CSV
	submission.to_csv(OUTPUT_CSV, index=False)
	print(f"Submission file saved to {OUTPUT_CSV}")
	print(f"First 5 rows:\n{submission.head()}")

if __name__ == "__main__":
	main()