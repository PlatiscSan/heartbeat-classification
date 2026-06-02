"""
Heartbeat Signal Classification - Data Analysis and Preprocessing
Corresponding to Chapter 2 of the report.
All outputs are saved to the 'output' directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Create output directory if it does not exist
OUTPUT_DIR = "output"

def parse_signals(signal_str):
	"""Convert comma-separated string to numpy array"""
	return np.array([float(x) for x in signal_str.split(',')])

def standardize(signal):
	"""Z-score standardization for a single signal"""
	mu = signal.mean()
	sigma = signal.std()
	if sigma < 1e-6:
		return signal - mu
	return (signal - mu) / sigma

def main():

	os.makedirs(OUTPUT_DIR, exist_ok=True)

	# ==================== 2.2 Data Loading ====================
	print("=" * 50)
	print("2.2 Data Loading")
	df = pd.read_csv('./datasets/train.csv')  # make sure the file path is correct
	print(f"Dataset shape: {df.shape}")

	# ==================== 2.3 Data Overview and Quality Check ====================
	print("\n2.3 Data Overview and Quality Check")
	print("Missing value statistics:")
	print(df.isnull().sum())

	print("\nSignal length verification:")
	lengths = df['heartbeat_signals'].apply(lambda x: len(x.split(',')))
	print(lengths.value_counts())

	label_counts = df['label'].value_counts().sort_index()
	print("\nLabel distribution:")
	print(label_counts)

	# Plot label distribution (bar + pie) -> saved to output/
	plt.figure(figsize=(12, 4))
	plt.subplot(1, 2, 1)
	label_counts.plot(kind='bar', color='skyblue')
	plt.title('Label Distribution (Bar)')
	plt.xlabel('Label')
	plt.ylabel('Count')
	plt.subplot(1, 2, 2)
	label_counts.plot(kind='pie', autopct='%1.1f%%', explode=[0.02]*4)
	plt.title('Label Distribution (Pie)')
	plt.tight_layout()
	plt.savefig(os.path.join(OUTPUT_DIR, 'label_distribution.png'), dpi=150)
	plt.close()
	print(f"Label distribution plot saved to {OUTPUT_DIR}/label_distribution.png")

	# ==================== 2.4 Signal Visualization ====================
	print("\n2.4 Signal Visualization")

	# Single class signal examples
	fig, axes = plt.subplots(2, 2, figsize=(12, 8))
	for i, ax in enumerate(axes.flat):
		sample = df[df['label'] == i].iloc[0]
		signal = parse_signals(sample['heartbeat_signals'])
		ax.plot(signal, linewidth=1)
		ax.set_title(f'Label {i} (Class {i})')
		ax.set_xlabel('Time point')
		ax.set_ylabel('Amplitude')
	plt.tight_layout()
	plt.savefig(os.path.join(OUTPUT_DIR, 'single_class_signals.png'), dpi=150)
	plt.close()
	print(f"Single class signals saved to {OUTPUT_DIR}/single_class_signals.png")

	# Mean signal and standard deviation per class
	print("\nComputing mean signals and standard deviations per class...")
	mean_signals = []
	std_signals = []
	for label in range(4):
		signals = df[df['label'] == label]['heartbeat_signals'].apply(parse_signals)
		signals_array = np.stack(signals.values)
		mean_signals.append(signals_array.mean(axis=0))
		std_signals.append(signals_array.std(axis=0))

	plt.figure(figsize=(12, 6))
	colors = ['blue', 'orange', 'green', 'red']
	for i in range(4):
		plt.plot(mean_signals[i], label=f'Class {i}', color=colors[i])
		plt.fill_between(range(205), mean_signals[i] - std_signals[i],
						mean_signals[i] + std_signals[i], alpha=0.2, color=colors[i])
	plt.legend()
	plt.title('Mean Signal with ±1 Std per Class')
	plt.xlabel('Time point')
	plt.ylabel('Amplitude')
	plt.tight_layout()
	plt.savefig(os.path.join(OUTPUT_DIR, 'mean_std_signals.png'), dpi=150)
	plt.close()
	print(f"Mean and std plot saved to {OUTPUT_DIR}/mean_std_signals.png")

	# ==================== 2.5 Data Preprocessing ====================
	print("\n2.5 Data Preprocessing")

	print("Parsing all signals...")
	X = np.array([parse_signals(s) for s in df['heartbeat_signals']])
	y = df['label'].values
	print(f"Feature matrix shape: {X.shape}, Label shape: {y.shape}")

	print("Applying Z-score standardization...")
	X = np.array([standardize(signal) for signal in X])
	print("Standardization done.")

	print("Splitting training and test sets (8:2, stratified)...")
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=y
	)
	print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

	print("\nTraining set label distribution:")
	train_counts = pd.Series(y_train).value_counts().sort_index()
	print(train_counts)
	print("Test set label distribution:")
	test_counts = pd.Series(y_test).value_counts().sort_index()
	print(test_counts)

	print("\nReshaping data for PyTorch Conv1d...")
	X_train = X_train[:, np.newaxis, :]   # (n, 1, 205)
	X_test = X_test[:, np.newaxis, :]
	print(f"Final training set shape: {X_train.shape}")
	print(f"Final test set shape: {X_test.shape}")

	# Save preprocessed data to output directory
	np.savez(os.path.join(OUTPUT_DIR, 'preprocessed_heartbeat.npz'),
			X_train=X_train, y_train=y_train,
			X_test=X_test, y_test=y_test)
	print(f"\nPreprocessed data saved to {OUTPUT_DIR}/preprocessed_heartbeat.npz")

	print("\n" + "=" * 50)
	print("Data analysis and preprocessing completed!")
	
if __name__ == "__main__":
	main()