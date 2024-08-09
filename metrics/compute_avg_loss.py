import torch
from typing import Callable

def compute_avg_loss(model: torch.nn.Module, 
                     loader: torch.utils.data.DataLoader, 
                     loss_fn: Callable,
                     device: torch.device = torch.device('cuda')) -> float:
    """
    Calculate the average loss over a Dataloader.

    Args:
        model: The neural network model.
        loader: DataLoader providing the dataset.
        loss_fn: Loss function to compute the loss.
        device: Device on which computations are performed.

    Returns:
        avg_loss: The average loss over the dataset.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    return avg_loss