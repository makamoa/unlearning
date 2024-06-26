import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

class BaseDatasetHandler:
    def __init__(self, batch_size, validation_split, num_workers, data_dir, random_seed):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.num_workers = num_workers
        self.random_seed = random_seed

    def _create_dataloaders(self, trainset, testset):
        # Create indices for training and validation splits
        num_train = len(trainset)
        indices = list(range(num_train))
        split = int(np.floor(self.validation_split * num_train))

        # Fix random seed for reproducibility
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, sampler=train_sampler, num_workers=self.num_workers)

        validloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, sampler=valid_sampler, num_workers=self.num_workers)

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return trainloader, validloader, testloader

    def get_transformations(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_dataloaders(self):
        raise NotImplementedError("This method should be overridden by subclasses")

class CIFAR100Handler(BaseDatasetHandler):
    def __init__(self, batch_size=64, validation_split=0.1, num_workers=2, data_dir='./data/cifar100', random_seed=None):
        super().__init__(batch_size, validation_split, num_workers, data_dir, random_seed)

    def get_transformations(self):
        # Define the transform for the training and testing data
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        return transform_train, transform_test

    def get_dataloaders(self):
        transform_train, transform_test = self.get_transformations()
        # Download and load the CIFAR-100 training dataset
        trainset = torchvision.datasets.CIFAR100(
            root=self.data_dir, train=True, download=True, transform=transform_train)

        # Download and load the CIFAR-100 test dataset
        testset = torchvision.datasets.CIFAR100(
            root=self.data_dir, train=False, download=True, transform=transform_test)

        return self._create_dataloaders(trainset, testset)
    
def get_dataloaders(dataset='cifar100', **kwargs):
    if dataset == 'cifar100':
        handler = CIFAR100Handler(**kwargs)
    else:
        raise ValueError(f"Dataset {dataset} is not supported")
    
    return handler.get_dataloaders()

if __name__ == '__main__':
    # Example usage
    trainloader, validloader, testloader = get_dataloaders(dataset='cifar100', batch_size=128, data_dir='./media/cifar100')
    print(trainloader.__len__())
