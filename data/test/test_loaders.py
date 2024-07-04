import pytest
import torch

from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import os

from data import BaseDatasetHandler, uniform_confuser, mix_both_sets, AmendedDatasetHandler

class RandomVisionDataset(VisionDataset):
    def __init__(self,
                 root,
                 num_samples=100,
                 num_classes=5,
                 image_size=(28, 28),
                 transform=None,
                 target_transform=None
                 ):
        super(RandomVisionDataset, self).__init__(root,
                                                  transform=transform,
                                                  target_transform=\
                                                    target_transform)
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.data, self.targets = self._generate_random_data()

    def _generate_random_data(self):
        data = []
        targets = []
        for _ in range(self.num_samples):
            img = np.random.randint(0, 256, (3,) + self.image_size,
                                    dtype=np.uint8)
            target = np.random.randint(0, self.num_classes)
            data.append(img)
            targets.append(target)
        return data, targets

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        target = self.targets[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def create_random_datasets(root,
                           num_train=1000,
                           num_test=200,
                           image_size=(28, 28)):
    transform = None

    train_dataset = RandomVisionDataset(root=root,
                                        num_samples=num_train,
                                        image_size=image_size,
                                        transform=transform)
    test_dataset = RandomVisionDataset(root=root,
                                       num_samples=num_test,
                                       image_size=image_size,
                                       transform=transform)

    return train_dataset, test_dataset

class RandomDatasetHandler(BaseDatasetHandler):
    def __init__(self, root, num_train, num_test, batch_size,
                 validation_split, num_workers):
        super().__init__(batch_size, validation_split, num_workers,
                         data_dir=None, random_seed=None)
        self.root = root
        self.num_train = num_train
        self.num_test = num_test

    def create_transformations(self):
        return ToTensor(), ToTensor()

    def get_datasets(self):
        return create_random_datasets(self.root,
                                      self.num_train,
                                      self.num_test)

def test_indices_do_not_intersect():
    dataset_handler = RandomDatasetHandler(root='./data/random_data',
                                           num_train=50,
                                           num_test=5,
                                           batch_size=32,
                                           validation_split=0.2,
                                           num_workers=2)
    data_confuser = uniform_confuser(confuse_level=.0, random_seed=42)
    splitter = mix_both_sets(amend_split=1., retain_split=0.1, random_seed=42)
    confused_dataset_handler = AmendedDatasetHandler(dataset_handler,
                                                     data_confuser,
                                                     splitter,
                                                     same_indices_for_unseen=False,
                                                     class_wise_corr=True
                                                     )
    train_loader, val_loader, test_loader, forget_loader, \
        retain_loader, unseen_loader = \
        confused_dataset_handler.get_dataloaders()
    
    train_idx = train_loader.sampler.indices
    val_idx = val_loader.sampler.indices
    forget_idx = forget_loader.sampler.indices
    retain_idx = retain_loader.sampler.indices
    unseen_idx = unseen_loader.sampler.indices

    all_idx = [val_idx, forget_idx, retain_idx, unseen_idx]
    count = 0
    for i in range(len(all_idx)):
        for j in range(i + 1, len(all_idx)):
            count += 1
            assert len(set(all_idx[i]).intersection(set(all_idx[j]))) == 0, \
                "Indices must not coincide"

    assert count == len(all_idx) * (len(all_idx) - 1) / 2
    assert len(set(train_idx)) == len(set(forget_idx).union(set(retain_idx)))