from typing import Any, Callable, List, Optional, Tuple, Union, Dict

import torch
import numpy as np

def class_to_indices_mapper(dataset: torch.utils.data.Dataset) -> Dict:
    """
    Map each class in a given dataset to the indices of samples
    belonging to that class.

    Args:
        dataset (torch.utils.data.Dataset): The dataset from which to
            extract class indices.
        
    Returns:
        cls_to_idx: A dictionary where the keys are the classes and the
        values are arrays of indices corresponding to the samples of
        each class.
    """
    targets = np.array(dataset.targets)
    classes = list(set(targets))
    classes = np.array(sorted(classes))
    assert np.array_equal(classes, np.arange(len(classes))), \
        "Classes do not match expected range"
    cls_to_idx = {cls: np.flatnonzero(targets == cls) for cls in classes}
    return cls_to_idx

def generate_random_list(upper_limit: int,
                         excluded_number: int,
                         count: int) -> np.array:
    """
    Generate a list of `count` random integers between 0 and `upper_limit`-1,
    excluding `excluded_number`.
    
    Args:
        upper_limit: Upper limit of the range (exclusive).
        excluded_number (int): The number to exclude.
        count (int): The number of random integers to generate.
    
    Returns:
        numpy.array: An array of `count` random integers
            between 0 and `upper_limit`-1, excluding `excluded_number`.
    """
    if excluded_number < 0 or excluded_number >= upper_limit:
        raise ValueError(
            "excluded_number must be within the range 0 to upper_limit-1"
            )

    valid_numbers = np.array([i for i in range(upper_limit)
                              if i != excluded_number])
    
    return np.random.choice(valid_numbers, count, replace=True)

def uniform_confuser(confuse_level=0.1, validation_split=0.2,
                     random_seed=None) -> Callable:
    """
    Create a function that amends a dataset by introducing a uniform
    level of confusion into the training data.

    Args:
        confuse_level: The proportion of training data to be
            confused by altering their class labels.
        validation_split: The proportion of the dataset to be
            used for validation.
        random_seed: Seed for the random number generator to ensure
            reproducibility.

    Returns:
        function: A function that takes a dataset as input and returns
            a tuple containing:
                  - The amended dataset with some class labels altered.
                  - A list of indices of the amended training samples.
                  - A list of indices of the retained training samples.
                  - A list of indices of the validation samples.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    def amend_function(dataset):
        cls_to_idx = class_to_indices_mapper(dataset)
        val_idx = np.array([], dtype=int)
        amended_train_idx = np.array([], dtype=int)
        retain_train_idx = np.array([], dtype=int)
        n_classes = len(cls_to_idx.keys())
        for target, indices in cls_to_idx.items():
            np.random.shuffle(indices)
            current_length = len(indices)
            # Define split indices for validation and amendment
            val_split_idx = int((1 - validation_split) * current_length)
            confuse_split_idx = int(val_split_idx * confuse_level)
            assert confuse_split_idx <= val_split_idx
            val_idx = np.concatenate(
                (val_idx, indices[val_split_idx:]))
            amended_train_idx = np.concatenate(
                (amended_train_idx, indices[:confuse_split_idx]))
            retain_train_idx = np.concatenate(
                (retain_train_idx, indices[confuse_split_idx:val_split_idx]))
            # `confusion targets` is an array of new, confused labels;
            # Since `confuse_split_idx` represents also the number of
            # confused indices, we utilize it as the `count` argument for 
            # the `generate_random_list` function.
            confusion_targets = generate_random_list(
                n_classes,
                target,
                confuse_split_idx)
            # Update the targets for the indices in amended_train_idx
            for counter, idx in enumerate(indices[:confuse_split_idx]):
                dataset.targets[idx] = confusion_targets[counter]
        return dataset, amended_train_idx, retain_train_idx, val_idx
    return amend_function