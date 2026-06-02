"""
Heartbeat Signal Classification - Training Script
Trains a 1D-CNN model using preprocessed data.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

# ==================== Configuration ====================
DATA_PATH = "output/preprocessed_heartbeat.npz"
MODEL_SAVE_PATH = "output/best_model.pth"
PLOT_SAVE_PATH = "output/training_curves.png"
LOG_PATH = "output/training_log.txt"

RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 10		   # Early stopping patience

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Set random seed for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==================== Dataset Class ====================
class HeartbeatDataset(Dataset):
	def __init__(self, X, y):
		self.X = torch.tensor(X, dtype=torch.float32)
		self.y = torch.tensor(y, dtype=torch.long)

	def __len__(self):
		return len(self.y)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]

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

		# Compute flattened size after conv + pool
		# Input length 205 -> after 3 pools: 205 // 2 // 2 // 2 = 25
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

def compute_class_weights(y_train):
	"""Compute class weights inversely proportional to class frequencies."""
	# Convert to integer type explicitly
	y_train = y_train.astype(np.int64)
	class_counts = np.bincount(y_train)
	total = len(y_train)
	num_classes = len(class_counts)
	weights = total / (num_classes * class_counts)
	return torch.tensor(weights, dtype=torch.float32)

# ==================== Training Function ====================
def train_epoch(model, loader, criterion, optimizer, device):
	model.train()
	running_loss = 0.0
	all_preds = []
	all_labels = []

	for inputs, labels in loader:
		inputs, labels = inputs.to(device), labels.to(device)

		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item() * inputs.size(0)
		preds = torch.argmax(outputs, dim=1)
		all_preds.extend(preds.cpu().numpy())
		all_labels.extend(labels.cpu().numpy())

	epoch_loss = running_loss / len(loader.dataset)
	epoch_acc = accuracy_score(all_labels, all_preds)
	return epoch_loss, epoch_acc

def validate_epoch(model, loader, criterion, device):
	model.eval()
	running_loss = 0.0
	all_preds = []
	all_labels = []

	with torch.no_grad():
		for inputs, labels in loader:
			inputs, labels = inputs.to(device), labels.to(device)
			outputs = model(inputs)
			loss = criterion(outputs, labels)

			running_loss += loss.item() * inputs.size(0)
			preds = torch.argmax(outputs, dim=1)
			all_preds.extend(preds.cpu().numpy())
			all_labels.extend(labels.cpu().numpy())

	epoch_loss = running_loss / len(loader.dataset)
	epoch_acc = accuracy_score(all_labels, all_preds)
	return epoch_loss, epoch_acc

# ==================== Main ====================
def main():
	# Load preprocessed data
	print("Loading preprocessed data...")
	data = np.load(DATA_PATH)
	X_train = data['X_train']
	y_train = data['y_train'].astype(np.int64)   # force integer
	X_test = data['X_test']
	y_test = data['y_test'].astype(np.int64)
	print(f"Train: {X_train.shape}, Test: {X_test.shape}")

	# Create datasets and loaders
	train_dataset = HeartbeatDataset(X_train, y_train)
	test_dataset = HeartbeatDataset(X_test, y_test)
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

	# Model, loss, optimizer
	model = HeartbeatCNN(num_classes=4).to(DEVICE)
	class_weights = compute_class_weights(y_train).to(DEVICE)
	criterion = nn.CrossEntropyLoss(weight=class_weights)
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

	# Training loop with early stopping
	best_val_loss = float('inf')
	best_epoch = -1
	patience_counter = 0

	train_losses, val_losses = [], []
	train_accs, val_accs = [], []

	print("\nStarting training...")
	for epoch in range(1, EPOCHS + 1):
		train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
		val_loss, val_acc = validate_epoch(model, test_loader, criterion, DEVICE)

		train_losses.append(train_loss)
		val_losses.append(val_loss)
		train_accs.append(train_acc)
		val_accs.append(val_acc)

		print(f"Epoch {epoch:2d}/{EPOCHS} | "
			  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
			  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

		# Learning rate scheduling
		scheduler.step(val_loss)

		# Early stopping and model saving
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_epoch = epoch
			patience_counter = 0
			torch.save(model.state_dict(), MODEL_SAVE_PATH)
			print(f"  -> New best model saved (val_loss = {val_loss:.4f})")
		else:
			patience_counter += 1
			if patience_counter >= PATIENCE:
				print(f"Early stopping triggered after {epoch} epochs.")
				break

	print(f"\nTraining completed. Best model at epoch {best_epoch} (val_loss = {best_val_loss:.4f})")

	# Plot training curves
	plt.figure(figsize=(12, 4))
	plt.subplot(1, 2, 1)
	plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
	plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.title('Training and Validation Loss')

	plt.subplot(1, 2, 2)
	plt.plot(range(1, len(train_accs)+1), train_accs, label='Train Accuracy')
	plt.plot(range(1, len(val_accs)+1), val_accs, label='Validation Accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.title('Training and Validation Accuracy')
	plt.tight_layout()
	plt.savefig(PLOT_SAVE_PATH, dpi=150)
	plt.close()
	print(f"Training curves saved to {PLOT_SAVE_PATH}")

	# Write log
	with open(LOG_PATH, 'w') as f:
		f.write(f"Best epoch: {best_epoch}\n")
		f.write(f"Best validation loss: {best_val_loss:.6f}\n")
		f.write(f"Final train loss: {train_losses[-1]:.6f}\n")
		f.write(f"Final validation loss: {val_losses[-1]:.6f}\n")
	print(f"Training log saved to {LOG_PATH}")
	
if __name__ == "__main__":
	main()