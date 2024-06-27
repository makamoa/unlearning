import torch
import numpy as np

def class_to_indices_mapper(dataset: torch.utils.data.Dataset) -> dict:
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
    assert classes == np.arange(len(classes)) # expected functionality
    cls_to_idx = {cls: np.flatnonzero(targets == cls) for cls in classes}
    return cls_to_idx

def uniform_confuser(confuse_level=0.1, validation_split=0.2,
                     random_seed=None):
    """
    Create a function that amends a dataset by introducing a uniform
    level of confusion into the training data.

    Args:
        confuse_level (float): The proportion of training data to be
            confused by altering their class labels. Default is 0.1.
        validation_split (float): The proportion of the dataset to be
            used for validation. Default is 0.2.
        random_seed (int, optional): Seed for the random number
            generator to ensure reproducibility. Default is None.

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
        val_idx = []
        amended_train_idx = []
        retain_train_idx = []
        n_classes = len(cls_to_idx.keys())
        for target, indices in cls_to_idx.items():
            np.random.shuffle(indices)
            current_length = len(indices)
            val_split_idx = current_length - \
                int(current_length * validation_split)
            confuse_split_idx = int(val_split_idx * confuse_level)
            val_idx.extend(indices[val_split_idx:])
            amended_train_idx.extend(indices[:confuse_split_idx])
            retain_train_idx.extend(
                indices[confuse_split_idx:val_split_idx])
            dataset[amended_train_idx] = target + 1 % n_classes
        return dataset, amended_train_idx, retain_train_idx, val_idx
    return amend_function