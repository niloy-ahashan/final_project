# Communication-Efficient Federated Learning with Count Mean Linear Sketch

## Overview
This repository implements a communication-efficient federated learning framework using sketch-based gradient compression. Instead of transmitting full gradient updates, participating clients compress their locally computed gradients using a Count Mean Sketch before sending them to a central aggregator. This approach significantly reduces communication overhead while preserving model performance.

The framework is evaluated on an image recognition task using deep neural network models with approximately 6.5 million parameters. Experiments are conducted on non-i.i.d. datasets consisting of 60,000 color images to reflect realistic federated learning scenarios.

## Project Structure
- `fed_aggregator.py` — Central server logic for aggregating gradients  
- `fed_worker.py` — Client-side training and gradient compression  
- `cv_train.py` — Image classification training pipeline  
- `utils.py` — Utility functions and argument parsing  
- `sketch/` — Implementation of Count Mean Sketch  
- `environment.yml` — Conda environment specification  

## Requirements
This project is implemented in **Python** and uses **PyTorch** for model training and distributed execution.

### Dependencies
- Python 3.8+
- PyTorch
- torchvision
- numpy
- scipy
- tqdm

All dependencies are specified in `environment.yml`.

## Setup Instructions
Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate fl-sketch
