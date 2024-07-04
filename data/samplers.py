import torch
import numpy as np
from data import class_to_indices_mapper
from copy import deepcopy

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

class _ClassWiseDependentSampler(torch.utils.data.Sampler):
    """
    Samples elements randomly from a given list of indices class-wise
    correspondingly with another sampler.
    
    Args:
    dependent_indices (Sequence[int]): The indices of the dependent samples.
    dataset (Sized): The dataset from which to sample.
    independent_sampler (torch.utils.data.Sampler): The independent sampler.
    """
    def __init__(self,
                 dependent_indices: Sequence[int],
                 dataset: Sized,
                 independent_sampler: torch.utils.data.Sampler
                 ):
        self.dependent_cls_to_idx_map = class_to_indices_mapper(dataset)
        for cls_ in self.dependent_cls_to_idx_map.keys():
            self.dependent_cls_to_idx_map[cls_] = \
                [x for x in 
                 dependent_indices 
                 if x in 
                 self.dependent_cls_to_idx_map[cls_]]
        self.indices = dependent_indices
        self.labels = dataset.targets
        self.independent_sampler = independent_sampler
        
    def __iter__(self) -> Iterator[int]:
        for ind in self.independent_sampler:
            if isinstance(self.labels, torch.Tensor):
                ind_cls = self.labels[ind].item()
            else:
                ind_cls = self.labels[ind]
            indices = self.dependent_cls_to_idx_map[ind_cls]
            assert len(indices) > 0, \
                "Indices of an expected class are missing"
            meta_ind = np.random.randint(low=0, high=len(indices))
            yield indices[meta_ind]

    def __len__(self) -> int:
        return len(self.indices)
    
class CorrespondingSubsetRandomSamplers:
    def __init__(self,
                 independent_indices: Sequence[int],
                 dependent_indices: Sequence[int],
                 dataset: Sized,
                 device: Optional[str]='cpu',
                 seed: int=42
                 ):
        assert device == 'cpu' or device == 'cuda'
        generator_1 = torch.Generator(device=device)
        generator_2 = torch.Generator(device=device)
        generator_1.manual_seed(seed)
        generator_2.manual_seed(seed)
        self.independent_sampler = torch.utils.data.sampler.SubsetRandomSampler(independent_indices, generator_1)
        self.copy_sampler = torch.utils.data.sampler.SubsetRandomSampler(independent_indices, generator_2)
        self.dependent_sampler = _ClassWiseDependentSampler(dependent_indices, dataset, self.copy_sampler)
    
    def return_samplers(self):
        return self.independent_sampler, self.dependent_sampler