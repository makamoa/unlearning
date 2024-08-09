from typing import Any, Callable, Iterable, TypeVar, Generic, List, Optional, Union, Tuple

# Constants
DEFAULT_RETAIN_REFERENCE = 90.0
DEFAULT_EPSILON_CONDITION = 0.1

def unlearning_stopping_criterion(stopping_criterion_enabled: str,
                                  forget_loss: float,
                                  validation_loss: float
                                  ) -> bool:
    """
    Determine whether to stop training based on the unlearning criterion.

    Args:
        stopping_criterion_enabled: The type of stopping criterion.
        forget_loss: The current forget loss.
        validation_loss: The current validation loss.

    Returns:
        bool: True if training should stop, False otherwise.
    """    
    return stopping_criterion_enabled == 'unlearning' and \
        forget_loss >= validation_loss

def refining_stopping_criterion(stopping_criterion_enabled: str,
                                val_loss_decreased_at_previous_epoch: bool,
                                current_val_loss: float,
                                previous_val_loss: float,
                                epoch: int,
                                loss_condition: float = 0.,
                                epsilon_condition: float =
                                    DEFAULT_EPSILON_CONDITION
                                ) -> Tuple[bool, bool, float]:
    """
    Determine whether to stop training based on the refining criterion.

    Args:
        stopping_criterion_enabled: The type of stopping criterion.
        val_loss_decreased_at_previous_epoch: Whether validation loss
            decreased at the previous epoch.
        current_val_loss: The current validation loss.
        previous_val_loss: The validation loss at previous epoch.
        epoch: The current epoch number.
        loss_condition: The loss value in the condition part of constrained
            optimization problem.
        epsilon_condition: Tolerance for loss condition.
    
    Returns:
        A tuple. First value indicates whether to stop training (True) or
        execution must continue (False). Second and third values update
        `val_loss_decreased_at_previous_epoch` and `previous_val_loss` for
        the next epoch.
    """    
    stop_training = epoch >= 2 and \
        stopping_criterion_enabled == 'refining' and \
        current_val_loss >= previous_val_loss and \
        abs(loss_condition) <= epsilon_condition and \
        not val_loss_decreased_at_previous_epoch
    val_loss_decreased_at_previous_epoch = \
        current_val_loss <= previous_val_loss \
            if previous_val_loss is not None else True

    return (
        stop_training,
        val_loss_decreased_at_previous_epoch,
        current_val_loss
        )

def forget_forever_stopping_criterion(stopping_criterion_enabled: str,
                                      retain_accuracy: float,
                                      retain_reference: float =
                                        DEFAULT_RETAIN_REFERENCE
                                      )-> bool:
    """
    Determine whether to stop training based on the forget-forever criterion.

    Args:
        stopping_criterion_enabled: The type of stopping criterion.
        retain_accuracy: The retain set accuracy.
        retain_reference: The reference accuracy threshold.

    Returns:
        bool: True if training should stop, False otherwise.
    """    
    return stopping_criterion_enabled == 'forget-forever' and \
        abs(retain_accuracy) <= retain_reference


def confusion_resolution_simple_stopping_criterion(
        stopping_criterion_enabled: str,
        forget_loss: float,
        dumb_forget_loss: Optional[float]
        ) -> bool:
    """
    Determine whether to stop training based on the confusion resolution
    criterion.

    Args:
        stopping_criterion_enabled: The type of stopping criterion.
        forget_loss: The forget set loss.
        dumb_forget_loss: The forget set loss on randomly initialized model.

    Returns:
        bool: True if training should stop, False otherwise.
    """    
    return stopping_criterion_enabled == 'confusion' and \
        forget_loss >= dumb_forget_loss