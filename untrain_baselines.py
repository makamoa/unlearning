import torch
from torch import optim
from helper_functions import get_current_datetime_string
from metrics import calculate_accuracy
from train import untrain_model
from optimizer import KL_retain_loss, KL_forget_loss

from typing import Literal, Callable

def reinitialize_weights(layer):
    """Reinitialize weights of a layer if it has 'reset_parameters' method."""
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()

def catastrophic_forgetting(model: torch.nn.Module,
                            k: int,
                            retainloader: torch.utils.data.DataLoader,
                            forgetloader: torch.utils.data.DataLoader,
                            validloader: torch.utils.data.DataLoader,
                            num_epochs: int=10,
                            learning_rate: float=0.001,
                            log_dir: str='runs',
                            device: Literal['cuda', 'cpu']='cuda',
                            name_prefix: str=get_current_datetime_string()) -> None:
    """
    Freeze the first `k` layers of the model and fine-tune the rest using base_loss.

    Args:
        model: The neural network model to be fine-tuned.
        k: The number of layers to freeze.
        retainloader: DataLoader for the data to be retained.
        forgetloader: DataLoader for the "forget" dataset. This loader is not
            used during the untraining process. Instead, it is employed to
            align the number of epochs reported and report the accuracy on it.
            All epochs referenced in this context correspond to the epochs
            processed over the `forgetloader`.
        validloader: DataLoader for the validation data.
        base_loss: Loss function for the retained data.
        num_epochs: Number of epochs.
        learning_rate: Learning rate for the base optimizer.
        log_dir: Directory to save TensorBoard logs.
        device: Device to run the model on (e.g., 'cpu' or 'cuda').
        name_prefix: Prefix for naming logs and saved models.
    """
    print(f'Running Catastrophic Forgetting-k.')
    print(f'First {k} layers are to be frozen...')

    layers = list(model.children())

    assert type(k) == int, 'Number of layers must be an integer'

    # if `k`` is negative, only last `k` layers are unfrozen
    if k < 0:
        k = len(layers) + k

    name_prefix += f'_cf_{k}'

    for i in range(min(k, len(layers))):
        for param in layers[i].parameters():
            param.requires_grad = False

    print(f'First {k} layers are frozen!')

    untrain_model(model=model,
                  retainloader=retainloader,
                  forgetloader=forgetloader,
                  validloader=validloader,
                  retain_optimizer=None,
                  forget_optimizer=None,
                  retain_loss=None,
                  forget_loss=None,
                  num_epochs=num_epochs,
                  learning_rate=learning_rate,
                  log_dir=log_dir,
                  device=device,
                  name_prefix=name_prefix,
                  use_sam=False,
                  rho=None)
    
    for param in model.parameters():
        param.requires_grad = True

    print('Catastrophic forgetting complete, ALL layers are now unfrozen and trainable.')


def exact_unlearning(model: torch.nn.Module,
                     k: int,
                     retainloader: torch.utils.data.DataLoader,
                     forgetloader: torch.utils.data.DataLoader,
                     validloader: torch.utils.data.DataLoader,
                     num_epochs: int=10,
                     learning_rate: float=0.001,
                     log_dir: str='runs',
                     device: Literal['cuda', 'cpu']='cuda',
                     name_prefix: str=get_current_datetime_string()) -> None:
    """
    Freeze the first `k` layers of the model, reinitialize the rest, and
    fine-tune using base_loss.

    Args:
        model: The neural network model to be fine-tuned.
        k: The number of layers to freeze.
        retainloader: DataLoader for the data to be retained.
        forgetloader: DataLoader for the "forget" dataset. This loader is not
            used during the untraining process. Instead, it is employed to
            align the number of epochs reported and report the accuracy on it.
            All epochs referenced in this context correspond to the epochs
            processed over the `forgetloader`.
        validloader: DataLoader for the validation data.
        num_epochs: Number of epochs.
        learning_rate: Learning rate for the base optimizer.
        log_dir: Directory to save TensorBoard logs.
        device: Device to run the model on (e.g., 'cpu' or 'cuda').
        name_prefix: Prefix for naming logs and saved models.
    """
    print(f'Running Exact Unlearning with k={k}.')
    print(f'First {k} layers are to be frozen...')

    layers = list(model.children())

    # if `k`` is negative, only last `k` layers are unfrozen
    if k < 0:
        k = len(layers) + k

    name_prefix += f'_eu_{k}'    

    for i in range(min(k, len(layers))):
        for param in layers[i].parameters():
            param.requires_grad = False

    print(f'First {k} layers are frozen!')

    print(f'Reinitializing the remaining layers...')
    for i in range(min(k, len(layers)), len(layers)):
        layers[i].apply(reinitialize_weights)

    print('Remaining layers reinitialized!')

    untrain_model(model=model,
                  retainloader=retainloader,
                  forgetloader=forgetloader,
                  validloader=validloader,
                  retain_optimizer=None,
                  forget_optimizer=None,
                  retain_loss=None,
                  forget_loss=None,
                  num_epochs=num_epochs,
                  learning_rate=learning_rate,
                  log_dir=log_dir,
                  device=device,
                  name_prefix=name_prefix,
                  use_sam=False,
                  rho=None)
    
    for param in model.parameters():
        param.requires_grad = True

    print('Exact unlearning complete, ALL layers are now unfrozen')


def finetuning(model: torch.nn.Module,
               retainloader: torch.utils.data.DataLoader,
               forgetloader: torch.utils.data.DataLoader,
               validloader: torch.utils.data.DataLoader,
               num_epochs: int=10,
               learning_rate: float=0.001,
               log_dir: str='runs',
               device: Literal['cuda', 'cpu']='cuda',
               name_prefix: str=get_current_datetime_string()) -> None:
    """
    Fine-tune the entire model on the retain set.

    Args:
        model: The neural network model to be fine-tuned.
        retainloader: DataLoader for the data to be retained.
        forgetloader: DataLoader for the "forget" dataset. This loader is not
            used during the untraining process. Instead, it is employed to
            align the number of epochs reported and report the accuracy on it.
            All epochs referenced in this context correspond to the epochs
            processed over the `forgetloader`.        
        validloader: DataLoader for the validation data.
        num_epochs: Number of epochs.
        learning_rate: Learning rate for the base optimizer.
        log_dir: Directory to save TensorBoard logs.
        device: Device to run the model on (e.g., 'cpu' or 'cuda').
        name_prefix: Prefix for naming logs and saved models.
    """
    print(f'Running Fine-tuning.')

    name_prefix += '_finetuning'

    untrain_model(model=model,
                  retainloader=retainloader,
                  forgetloader=forgetloader,
                  validloader=validloader,
                  retain_optimizer=None,
                  forget_optimizer=None,
                  retain_loss=None,
                  forget_loss=None,
                  num_epochs=num_epochs,
                  learning_rate=learning_rate,
                  log_dir=log_dir,
                  device=device,
                  name_prefix=name_prefix,
                  use_sam=False,
                  rho=None)

    print('Fine-tuning complete.')


def neggradplus(model: torch.nn.Module,
                retainloader: torch.utils.data.DataLoader,
                forgetloader: torch.utils.data.DataLoader,
                validloader: torch.utils.data.DataLoader,
                forget_loss: Callable,
                forget_optimizer: optim,
                num_epochs: int=10,
                learning_rate: float=0.001,
                log_dir: str='runs',
                device: Literal['cuda', 'cpu']='cuda',
                name_prefix: str=get_current_datetime_string()) -> None:
    """
    Perform optimizer steps on the base_loss and then on the forget_loss.

    Args:
        model: The neural network model to be fine-tuned.
        retainloader: DataLoader for the data to be retained.
        forgetloader: DataLoader for the data to be forgotten.
        validloader: DataLoader for the validation data.
        base_loss: Base loss function for the retained data.
        forget_loss: Forget loss function for the forgotten data.
        num_epochs: Number of epochs.
        learning_rate: Learning rate for the base optimizer.
        log_dir: Directory to save TensorBoard logs.
        device: Device to run the model on (e.g., 'cpu' or 'cuda').
        name_prefix: Prefix for naming logs and saved models.
    """
    print('Running NegGrad+.')

    name_prefix += '_neggradplus'

    # Use the untrain_model function
    untrain_model(model=model,
                  retainloader=retainloader,
                  forgetloader=forgetloader,
                  validloader=validloader,
                  retain_optimizer=None,
                  forget_optimizer=forget_optimizer,
                  retain_loss=None,
                  forget_loss=forget_loss,
                  num_epochs=num_epochs,
                  learning_rate=learning_rate,
                  log_dir=log_dir,
                  device=device,
                  name_prefix=name_prefix,
                  use_sam=False,
                  rho=None)

    print('NegGrad+ complete.')


def SCRUB(model: torch.nn.Module,
          retainloader: torch.utils.data.DataLoader,
          forgetloader: torch.utils.data.DataLoader,
          validloader: torch.utils.data.DataLoader,
          retain_optimizer: optim.Optimizer,
          forget_optimizer: optim.Optimizer,
          forget_scheduler: torch.optim.lr_scheduler.StepLR,
          num_epochs: int=10,
          learning_rate: float=0.001,
          log_dir: str='runs',
          device: Literal['cuda', 'cpu']='cuda',
          name_prefix: str=get_current_datetime_string()) -> None:
    """
    Perform unlearning using the SCRUB method.

    Args:
        model: The neural network model to be fine-tuned.
        retainloader: DataLoader for the data to be retained.
        forgetloader: DataLoader for the data to be forgotten.
        validloader: DataLoader for the validation data.
        retain_optimizer: Optimizer for the retained data.
        forget_optimizer: Optimizer for the forgotten data.
        forget_scheduler: Scheduler for controlling the learning rate of
            the `forget_optimizer` optimizer.
        num_epochs: Number of epochs.
        learning_rate: Learning rate for the base optimizer.
        log_dir: Directory to save TensorBoard logs.
        device: Device to run the model on (e.g., 'cpu' or 'cuda').
        name_prefix: Prefix for naming logs and saved models.
    """
    print(f'Running SCRUB...')

    name_prefix += '_SCRUB' # to differentiate the runs

    assert forget_scheduler is not None, "Forget scheduler must be defined"

    untrain_model(model=model,
                  retainloader=retainloader,
                  forgetloader=forgetloader,
                  validloader=validloader,
                  retain_optimizer=retain_optimizer,
                  forget_optimizer=forget_optimizer,
                  retain_loss=KL_retain_loss,
                  forget_loss=KL_forget_loss,
                  num_epochs=num_epochs,
                  learning_rate=learning_rate,
                  log_dir=log_dir,
                  device=device,
                  name_prefix=name_prefix,
                  scheduler=forget_scheduler,
                  use_sam=False,
                  rho=None)

    print('SCRUB complete.')