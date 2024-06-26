import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

class BaseDatasetHandler:
    def __init__(self, batch_size, validation_split, num_workers, data_dir, random_seed):
        assert validation_split >= 0. and validation_split < 1.
        assert batch_size >= 1
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

        trainloader = self._create_trainloader(trainset, train_sampler)
        validloader = self._create_trainloader(trainset, valid_sampler)
        testloader = self._create_testloader(testset)

        return trainloader, validloader, testloader
    
    def _create_trainloader(self, dataset, sampler):
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)
    
    def _create_testloader(self, dataset):
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def create_transformations(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def get_datasets(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    def get_dataloaders(self):
        raise NotImplementedError("This method should be overridden by subclasses")


class CIFAR100Handler(BaseDatasetHandler):
    def __init__(self, batch_size=64, validation_split=0.1, num_workers=2, data_dir='./data/cifar100', random_seed=None):
        super().__init__(batch_size, validation_split, num_workers, data_dir, random_seed)

    def create_transformations(self):
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
    
    def get_datasets(self):
        transform_train, transform_test = self.create_transformations()
        # Download and load the CIFAR-100 training dataset
        trainset = torchvision.datasets.CIFAR100(
            root=self.data_dir, train=True, download=True, transform=transform_train)

        # Download and load the CIFAR-100 test dataset
        testset = torchvision.datasets.CIFAR100(
            root=self.data_dir, train=False, download=True, transform=transform_test)
        
        return trainset, testset

    def get_dataloaders(self):
        trainset, testset = self.get_datasets()
        return self._create_dataloaders(trainset, testset)
    

class AmendedDatasetHandler(BaseDatasetHandler):
    def __init__(self, dataset_handler, amend_function, 
                 forget_retain_function, random_seed=None):
        """
        Initializes a handler for an amended dataset, enabling experiments 
        with modified data points, such as confused labels or deleted classes.

        Args:
            dataset_handler (BaseDatasetHandler): An instance of a subclass 
                of BaseDatasetHandler responsible for managing the base dataset.
            amend_function (function): A function that takes a dataset and 
                returns the amended dataset, along with the indices of the 
                amended and retained data points in the train set, and the 
                indices in the validation set.
            forget_retain_function (function): A function that takes 
                the indices of the amended and retained data points in the 
                train set and returns the indices to be used for 
                the forgetloader and retainloader.
            random_seed (int, optional): A seed for random number generation 
                to ensure reproducibility. Default is None.

        Methods:
            _amend_dataset(dataset): Applies the amend_function to 
                the dataset.
            get_dataloaders(): Generates dataloaders for the amended train,
                potentially amended validation, and test datasets, as well as 
                for the forgetloader and retainloader.
        """
        super().__init__(dataset_handler.batch_size,
                         dataset_handler.validation_split,
                         dataset_handler.num_workers,
                         dataset_handler.data_dir,
                         random_seed)
        self.dataset_handler = dataset_handler
        self.amend_function = amend_function
        self.forget_retain_function = forget_retain_function

    def _amend_dataset(self, dataset):
        return self.amend_function(dataset)
    
    def get_dataloaders(self):
        trainset, testset = self.dataset_handler.get_datasets()

        amended_trainset, amended_train_idx, retained_train_idx, val_idx = \
            self._amend_dataset(trainset)

        forget_idx, retained_idx = self.forget_retain_function(
            amended_train_idx, retained_train_idx)
        train_idx = np.concatenate(amended_train_idx, retained_train_idx)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(val_idx)
        forget_sampler = SubsetRandomSampler(forget_idx)
        retain_sampler = SubsetRandomSampler(retained_idx)

        trainloader = self._create_trainloader(amended_trainset,
                                               train_sampler)
        validloader = self._create_trainloader(amended_trainset,
                                               valid_sampler)
        forgetloader = self._create_trainloader(amended_trainset,
                                                forget_sampler)
        retainloader = self._create_trainloader(amended_trainset,
                                                retain_sampler)
        testloader = self._create_testloader(testset)

        return trainloader, validloader, testloader, forgetloader, \
            retainloader        


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
