from typing import Any, Callable, List, Optional, Tuple, Union, Dict

import numpy as np

def simple_forget_retain_indices(amended_idx: np.array, retained_idx: np.array) -> Tuple[np.array, np.array]:
    """
    Return all amended indices for forget dataset and all retained
    indices for retain dataset.

    Args:
        amended_idx (np.array): An array of indices that have been
            amended in the original dataset.
        retained_idx (np.array): An array of indices that have not
            been amended in the original dataset.
    """
    return amended_idx, retained_idx

def mix_both_sets(amend_split: float, retain_split: float, 
                  random_seed: int=None
                  ) -> Callable:
    """
    Construct a function, generating a random mix of `amended_idx` and
    `retained_idx`.
    
    Args:
    <placeholder>
    """
    assert amend_split >=0 and amend_split <= 1.0, \
        "split value out of range"
    assert retain_split >=0 and retain_split <= 1.0, \
        "split value out of range"
    
    def mixer(amended_idx: np.array, retained_idx: np.array):
        if random_seed is not None:
            np.random.seed(random_seed)

        np.random.shuffle(amended_idx)
        np.random.shuffle(retained_idx)

        amend_split_idx = int(len(amended_idx) * amend_split)
        retain_split_idx = int(len(retained_idx) * retain_split)

        forget_1, retain_1 = \
            amended_idx[:amend_split_idx], amended_idx[amend_split_idx:]
        forget_2, retain_2 = \
            retained_idx[:retain_split_idx], retained_idx[retain_split_idx:]
        
        set_to_forget = np.concatenate((forget_1, forget_2))
        set_to_retain = np.concatenate((retain_1, retain_2))

        return set_to_forget, set_to_retain
    return mixer

if __name__ == "__main__":
    array_1 = np.arange(10)
    array_2 = np.arange(11, 20, 1)
    foo = mix_both_sets(0.1, 0.5, random_seed=2)
    print(foo(array_1, array_2))