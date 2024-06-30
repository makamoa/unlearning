import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def compute_loss_distribution_distance(model1, model2, dataloader, device='cuda'):
    """
    Computes the distance between the loss distributions of two models on a given dataset.

    :param model1: The first model.
    :param model2: The second model.
    :param dataloader: DataLoader for the dataset to evaluate the models on.
    :param device: Device to use for computation (e.g., 'cuda' or 'cpu').
    :return: The KL divergence between the loss distributions of the two models.
    """

    model1.to(device)
    model2.to(device)
    model1.eval()
    model2.eval()

    criterion = nn.CrossEntropyLoss(reduction='none')  # Reduction 'none' to get the individual losses
    losses1 = []
    losses2 = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs1 = model1(inputs)
            outputs2 = model2(inputs)

            loss1 = criterion(outputs1, labels)
            loss2 = criterion(outputs2, labels)

            losses1.extend(loss1.cpu().numpy())
            losses2.extend(loss2.cpu().numpy())

    # Convert lists to tensors
    losses1 = torch.tensor(losses1)
    losses2 = torch.tensor(losses2)

    # Compute histograms of the loss distributions
    hist1 = torch.histc(losses1, bins=50, min=losses1.min(), max=losses1.max())
    hist2 = torch.histc(losses2, bins=50, min=losses2.min(), max=losses2.max())

    # Normalize histograms to form probability distributions
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()

    # Add a small value to avoid log(0) issues
    epsilon = 1e-10
    hist1 = hist1 + epsilon
    hist2 = hist2 + epsilon

    # Compute KL divergence
    kl_divergence = F.kl_div(hist1.log(), hist2, reduction='sum')

    return kl_divergence.item()


def membership_inference_attack(model, train_loader, test_loader, device='cuda'):
    """
    Performs a membership inference attack on a given model using the provided training and testing data.

    :param model: The model to be attacked.
    :param train_loader: DataLoader for the training data.
    :param test_loader: DataLoader for the testing data.
    :param device: Device to use for computation (e.g., 'cuda' or 'cpu').
    :return: Dictionary with accuracy, precision, recall, and balanced accuracy of the membership inference attack.
    """

    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction='none')  # To get the individual losses

    # Collect losses for both training and testing data
    train_losses = []
    test_losses = []

    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_losses.extend(loss.cpu().numpy())

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_losses.extend(loss.cpu().numpy())

    # Convert lists to tensors
    train_losses = torch.tensor(train_losses)
    test_losses = torch.tensor(test_losses)

    # Concatenate train and test losses
    all_losses = torch.cat((train_losses, test_losses))

    # Create labels for membership inference (1 for training data, 0 for testing data)
    true_membership = torch.cat((torch.ones(len(train_losses)), torch.zeros(len(test_losses))))

    # Simple threshold-based attack: assume that lower loss means more likely to be a member
    threshold = all_losses.median()
    predicted_membership = (all_losses <= threshold).float()

    # Calculate metrics
    true_positive = ((predicted_membership == 1) & (true_membership == 1)).sum().item()
    false_positive = ((predicted_membership == 1) & (true_membership == 0)).sum().item()
    true_negative = ((predicted_membership == 0) & (true_membership == 0)).sum().item()
    false_negative = ((predicted_membership == 0) & (true_membership == 1)).sum().item()

    accuracy = (true_positive + true_negative) / len(true_membership)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
    balanced_accuracy = 0.5 * (
                true_positive / (true_positive + false_negative) + true_negative / (true_negative + false_positive))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'balanced_accuracy': balanced_accuracy
    }


if __name__ == '__main__':
    # Define transformations for the CIFAR-100 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Load the CIFAR-100 dataset
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


    # Define two simple models for testing
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(32 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 100)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate and load the models
    model1 = SimpleCNN().to(device)
    model2 = SimpleCNN().to(device)

    # For testing purposes, we use randomly initialized models
    # In practice, you would load trained models

    # Compute the KL divergence between the loss distributions of the two models
    kl_divergence = compute_loss_distribution_distance(model1, model2, testloader, device)
    print(f'KL Divergence between the loss distributions: {kl_divergence}')
