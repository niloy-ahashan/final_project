# Communication-Efficient Federated Learning with Count Mean Linear Sketch

## Overview
This repository implements a communication-efficient federated learning framework using sketch-based gradient compression. Instead of transmitting full gradient updates, participating clients compress their locally computed gradients using a Count Mean Sketch before sending them to a central aggregator. This approach significantly reduces communication overhead while preserving model performance.

The framework is evaluated on an image recognition task using deep neural network models with approximately 6.5 million parameters. Experiments are conducted on non-i.i.d. datasets consisting of 60,000 color images to reflect realistic federated learning scenarios.

## Project Structure
- `fed_aggregator.py` — Central server logic for aggregating gradients  
- `fed_worker.py` — Client-side training and gradient compression  
- `cv_train.py` — Image classification training pipeline  
- `utils.py` — Utility functions and argument parsing  
- `cmvec.py` — Implementation of Count Mean Sketch  

## Requirements
This project is implemented in **Python** and uses **PyTorch** for model training and distributed execution.

### Dependencies
- Python 3.8+
- PyTorch
- torchvision
- numpy
- scipy
- tqdm

---

## How to Run the Code

### 1. Run Federated Averaging (FedAvg Baseline)

```bash
python3 cv_train.py \
  --dataset_dir ~/datasets/cifar10/ \
  --dataset_name CIFAR10 \
  --model FixupResNet9 \
  --mode fedavg \
  --num_clients 10000 \
  --num_devices 1 \
  --num_workers 100 \
  --share_ps_gpu \
  --device cuda \
  --lr 0.06 \
  --num_epochs 12 \
  --local_batch_size -1 \
  --local_momentum 0.0 \
  --max_grad_norm 2.5
```

### 2. Run Count Mean Sketch–Based Federated Learning

```bash
python3 cv_train.py \
  --dataset_dir ~/datasets/cifar10/ \
  --dataset_name CIFAR10 \
  --model FixupResNet9 \
  --mode sketch \
  --local_batch_size 5 \
  --local_momentum 0.0 \
  --virtual_momentum 0.9 \
  --error_type virtual \
  --num_clients 10000 \
  --num_devices 1 \
  --num_workers 100 \
  --share_ps_gpu \
  --k 10000 \
  --num_rows 5 \
  --num_cols 500000 \
  --device cuda \
  --lr_scale 0.06 \
  --num_blocks 1
```



