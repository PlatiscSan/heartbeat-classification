#!/usr/bin/env python3
"""
run_all.py - Complete pipeline for heartbeat signal classification
Executes data preprocessing, model training, and evaluation sequentially.
"""

import sys
import subprocess
import os
from datetime import datetime

def run_script(script_name, description):
	"""Run a Python script and return success status."""
	print("\n" + "="*60)
	print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting: {description}")
	print("="*60)
	
	try:
		result = subprocess.run(
			[sys.executable, script_name],
			check=True,
			capture_output=False,
			text=True
		)
		print(f"\n{description} completed successfully.\n")
		return True
	except subprocess.CalledProcessError as e:
		print(f"\n{description} failed with exit code {e.returncode}.\n")
		print("Error output:")
		print(e.stderr if e.stderr else "See console output above.")
		return False
	except FileNotFoundError:
		print(f"\nScript '{script_name}' not found. Make sure it is in the current directory.\n")
		return False

def main():
	"""Run the complete pipeline."""
	print("\n" + "#"*60)
	print("#  Heartbeat Signal Classification Pipeline")
	print("#  Author: Experiment Report")
	print(f"#  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	print("#"*60)

	# Check if train.csv exists
	if not os.path.exists('./datasets/train.csv'):
		print("\n[ERROR] 'train.csv' not found.")
		sys.exit(1)

	# Step 1: Data Analysis & Preprocessing
	if not run_script('analyzation.py', 'Data Analysis and Preprocessing'):
		sys.exit(1)

	# Step 2: Model Training
	if not run_script('train.py', 'Model Training'):
		sys.exit(1)

	# Step 3: Model Evaluation
	if not run_script('evaluate.py', 'Model Evaluation'):
		sys.exit(1)

	# Final success message
	print("\n" + "#"*60)
	print(f"#  Pipeline completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	print("#  All results are saved in the 'output/' directory.")
	print("#  - preprocessed_heartbeat.npz")
	print("#  - best_model.pth")
	print("#  - training_curves.png")
	print("#  - confusion_matrix.png")
	print("#  - evaluation_report.txt")
	print("#  - training_log.txt")
	print("#"*60 + "\n")

if __name__ == "__main__":
	main()