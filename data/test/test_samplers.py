import pytest
import torch

from torch.utils.data import Dataset
from data import CorrespondingSubsetRandomSamplers, class_to_indices_mapper

class MockDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def test_samplers_correct_classes():
    data = torch.randn(200, 128)

    half_targets = torch.randint(0, 10, (100,))
    targets = torch.concatenate((half_targets, half_targets))

    dataset = MockDataset(data, targets)

    data_len = len(dataset)
    half_ind = int(data_len / 2)
    independent_indices = list(range(half_ind))
    dependent_indices = list(range(half_ind, data_len))
    
    samplers = CorrespondingSubsetRandomSamplers(
        independent_indices=independent_indices,
        dependent_indices=dependent_indices,
        dataset=dataset,
        device='cpu',
        seed=42
    )

    independent_sampler, dependent_sampler = samplers.return_samplers()

    assert len(independent_sampler) == len(independent_indices)
    assert len(dependent_sampler) == len(dependent_indices)

    count = 5 * len(dataset)

    indices_1 = []
    indices_2 = []

    for ind_1, ind_2 in zip(independent_sampler, dependent_sampler):
        indices_1.append(ind_1)
        indices_2.append(ind_2)
        if count < 0: break
        assert dataset.targets[ind_1].eq(dataset.targets[ind_2])
        count -= 1

    assert len(set(indices_1).intersection(set(indices_2))) == 0