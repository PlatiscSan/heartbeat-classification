# Heartbeat Signal Classification using 1D-CNN

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.13+](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains a complete solution for the **Tianchi Heartbeat Signal Classification** competition ([link](https://tianchi.aliyun.com/competition/entrance/531883/introduction)). A 1‑Dimensional Convolutional Neural Network (1D‑CNN) is trained to classify ECG heartbeat signals into four categories: normal, ventricular premature beat, supraventricular premature beat, and fusion beat.

The project covers the entire pipeline:
- Data analysis & preprocessing (standardization, train/test split)
- Model training with weighted cross‑entropy, dropout, learning rate scheduling, and early stopping
- Evaluation using both standard metrics (accuracy, F1‑score) and the official competition metric (**abs‑sum**)
- Generation of submission files in Tianchi format

---

## Table of Contents

- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Quick Start (One‑click Pipeline)](#quick-start-one-click-pipeline)
  - [Step‑by‑Step Execution](#step-by-step-execution)
- [Project Structure](#project-structure)
- [Results](#results)
- [Experiment Report](#experiment-report)
- [License](#license)

---

## Dataset

The original dataset `train.csv` contains **205‑point ECG heartbeat signals** and their corresponding labels (0,1,2,3).  

The original training data is randomly split into **training (80%)** and **test (20%)** using stratified sampling (`random_state=42`). All signals are Z‑score standardized.

---

## Requirements

- Python 3.9 or higher
- PyTorch 1.13+
- NumPy, Pandas, Matplotlib, Seaborn
- scikit‑learn

A **GPU** is recommended for faster training, but the code also runs on CPU.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/PlatiscSan/heartbeat-classification
   cd heartbeat-classification