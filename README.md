# CVNN: Complex-Valued Neural Networks for Computer Vision

## Installation

### Prerequisites
- Python 3.12+
- Poetry for dependency management

### Setup

1. **Clone and install dependencies**:
   ```bash
   git clone <repository-url>
   cd cvnn
   poetry install
   ```

2. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

3. **Verify installation**:
   ```bash
   python -m cvnn --help
   ```


## Quick Start

### Running Experiment

```bash
# Full pipeline (train + evaluate + visualize)
python -m cvnn configs/config_reconstruction.yaml --mode full

# Train only
python -m cvnn configs/config_reconstruction.yaml --mode train

# Evaluate existing model
python -m cvnn configs/config_reconstruction.yaml \
    --mode eval --resume-logdir logs/segmentation_experiment_*

# Resume training from checkpoint
python -m cvnn configs/config_reconstruction.yaml \
    --mode retrain --resume-logdir logs/reconstruction_*
```

## 🔧 Configuration System

### Configuration Hierarchy

1. **Base Configuration** (`configs/config.yaml`):
   ```yaml
   # Common settings for all tasks
   project_name: "cvnn_experiments"
   seed: 42
   
   logging:
     use_wandb: true
   
   training:
     epochs: 100
   ```

2. **Task-Specific Configuration** (`configs/config_reconstruction.yaml`):
   ```yaml
   # config_reconstruction.yaml
   task: "reconstruction"
   
   # Inherits from base config and adds:
   model:
     name: "AutoEncoder"
     layer_mode: complex
   ```


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [TorchCVNN](https://github.com/ivannz/torch-cvnn)
- Experiment tracking with [Weights & Biases](https://wandb.ai/)
---