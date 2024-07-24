import torch
from typing import Callable

def perform_optimizer_step(model: torch.nn,
                           model_teacher: torch.nn,
                           inputs: torch.Tensor,
                           labels: torch.Tensor,
                           loss_fn: Callable,
                           optimizer: torch.optim,
                           use_sam: bool):
    """
    Perform a single optimization step.

    Args:
        model: The neural network model.
        model_teacher: The teacher neural network model.
        inputs: Input batch
        labels: Ground truth labels
        loss_fn: Loss function that accepts model, model_teacher,
            inputs, labels as parameters
        optimizer: Optimizer.
        use_sam: Boolean to indicate whether to use SAM optimizer.

    Returns:
        loss_value: Calculated loss value.
    """
    if use_sam:
        loss_value = loss_fn(model, model_teacher, inputs, labels)
        loss_value.backward()
        optimizer.first_step(zero_grad=True)

        loss_value = loss_fn(model, model_teacher, inputs, labels)
        loss_value.backward()
        optimizer.second_step(zero_grad=True)
    else:
        loss_value = loss_fn(model, model_teacher, inputs, labels)
        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss_value