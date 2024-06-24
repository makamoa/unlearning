import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

def get_cifar100_dataloaders(batch_size=64, validation_split=0.1, num_workers=2, data_dir='./data/cifar100', random_seed=None):
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

    # Download and load the CIFAR-100 training dataset
    trainset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train)

    # Create indices for training and validation splits
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(validation_split * num_train))

    # Fix random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)

    validloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)

    # Download and load the CIFAR-100 test dataset
    testset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, validloader, testloader

if __name__ == '__main__':
    # Example usage
    trainloader, validloader, testloader = get_cifar100_dataloaders(batch_size=64, data_dir='/media/makarem/Data/cifar100')
    print(trainloader.__len__())
