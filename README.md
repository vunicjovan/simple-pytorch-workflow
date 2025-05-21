# simple-pytorch-workflow

A minimal Python project demonstrating a simple PyTorch workflow: synthetic data generation, train-test splitting, model training, saving/loading, and visualization with Matplotlib.

## Features

- Device-agnostic PyTorch code (CPU/GPU)
- Synthetic linear data generation
- Train/test split
- Model training and evaluation
- Model saving and loading
- Data visualization with Matplotlib
- Centralized logging
- Modular service-based architecture
- Dependency management with [UV](https://github.com/astral-sh/uv)

## Project Structure

- `main.py` — Main script: data generation, splitting, training, saving/loading, and plotting
- `models/linear_regression_model.py` — Simple linear regression model (PyTorch)
- `services/data_service.py` — Data generation and train/test split logic
- `services/model_service.py` — Model training, saving, and loading logic
- `services/visualization_service.py` — Data and prediction visualization
- `log/log_configuration.py` — Centralized logging configuration
- `pyproject.toml` — Project metadata and dependencies
- `uv.lock` — Lock file for reproducible installs
- `.gitignore` — Standard Python and JetBrains ignores
- `stored-models/` — Saved model files
- `logs/` — Log files

## Requirements

- Python >= 3.11
- [UV](https://github.com/astral-sh/uv) (for dependency management)

## Installation

1. Install [UV](https://github.com/astral-sh/uv) if you haven't already.
2. Install dependencies:
   ```sh
   uv sync

## Usage
Run the main script with:
```sh
uv run main.py
```

This will:

- Generate synthetic linear data
- Split it into training and test sets
- Train a linear regression model
- Plot the training data, test data, and predictions
- Save the trained model to `stored-models/linear_regression_model.pth`
- Load the model and evaluate it on the test set

## Logging

Logs are saved to `logs/main.log` and also printed to the console.
