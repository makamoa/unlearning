# Constrained Unlearning Project

## Summary

### Machine Unlearning: A Constraint-Based Approach

This project is inspired by the emerging human right "to be forgotten" and the need to remove unethical model outputs, which highlights the demand for resource-efficient machine unlearning techniques. We introduce a novel framework based on constrained optimization, aimed at both robust and efficient machine unlearning. By formulating unlearning as a constrained optimization problem, our approach selectively removes specific data points while effectively preserving the model's overall performance. Additionally, we propose a constrained fine-tuning algorithm that enhances model accuracy on retained data while ensuring low accuracy on the forget set, effectively separating performance between the two. Extensive experimentation on CIFAR-100 using ResNet demonstrates the superior unlearning precision and computational efficiency of our method compared to existing techniques like SCRUB and NegGrad+. We also offer theoretical insights into our problem formulation, further reinforcing the robustness and efficacy of this approach.

## Project Structure

The repository is organized as follows:

- `data/` - Contains dataset-related files and preprocessing scripts.
- `contrastive_discrimination/` - Directory for contrastive discrimination methods.
- `helper_functions/` - Helper functions for metrics, logging, and model utilities.
- `models/` - Model architectures, including batch averaging additions.
- `optimizer/` - Optimizer configurations.
- `runs/` - Directory for saving logs and checkpoints.
- `contrastive_classifiers.ipynb` - Jupyter notebook for contrastive classification experiments.
- `train.py` - Main training script.
- `train_constrained_unlearning.py` - Script for constrained unlearning on CIFAR-100.
- `train_contrastive.py` - Training script for contrastive models.
- `test.py` - Script for testing models.
- `environment.yml` - Conda environment file.
- `requirements.txt` - Python dependencies.
- `.gitignore` - Git ignore file.
- `README.md` - Project README file.


### Key Functions

- **`untrain_constrained()`**: Performs constrained unlearning by applying a penalty to ensure specific data (in `forgetloader`) is effectively forgotten, while retaining knowledge from other data (in `retainloader`). This function includes a training loop with gradient normalization and accuracy tracking.
- **`calculate_accuracy()`**: Calculates top-k accuracy metrics for given model outputs.
- **`_grad_norm()`**: Computes the gradient norm, which helps monitor and control gradient magnitudes during training.
- **`save_config()` and `load_config()`**: Functions for saving and loading YAML configurations.

## Setup and Usage

### Prerequisites

- Python 3.11
- PyTorch
- TensorBoard
- YAML library for configuration management

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/constrained_unlearning_project.git
    cd constrained_unlearning_project
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Training and Unlearning

To train the model and perform constrained unlearning, run:
```bash
python train_constrained_unlearning.py --num_epochs 10 --batch_size 32 --learning_rate 0.001 --data_dir './data/cifar100' --log_dir 'runs' --model 'resnet18' --device 'cuda' --untrain True --weight_path 'path_to_weights.pth'