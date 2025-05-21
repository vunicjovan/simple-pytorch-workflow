# simple-pytorch-workflow

A minimal Python project demonstrating a simple PyTorch workflow, including data generation, train-test splitting, model training, saving/loading, and visualization with Matplotlib.

## Features

- Device-agnostic PyTorch code (CPU/GPU)
- Synthetic linear data generation
- Train/test split
- Model training and evaluation
- Model saving and loading
- Data visualization with Matplotlib
- Logging setup
- Dependency management with [UV](https://github.com/astral-sh/uv)

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
```python
uv run main.py
```

This will generate synthetic data, split it into training and test sets, and display a plot of the data.
