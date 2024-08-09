import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from typing import Callable

def calculate_loss(model: torch.nn.Module,
                   dataloader: DataLoader,
                   loss_fn: Callable,
                   device: torch.device):
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            if loss.dim() == 0:
                losses.append(loss.item())
            else:
                losses += list(loss.cpu().detach().numpy())
    return np.array(losses)

def mia_efficacy(model: torch.nn.Module,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 forget_loader: DataLoader,
                 device: torch.device, 
                 loss_fn: Callable):
    """
    Compute the MIA efficacy metric defined in the 
    `Model Sparsity Can Simplify Machine Unlearning` Jia et al.
    """
    # Calculate losses for train, test, and forget sets
    train_losses = calculate_loss(model, train_loader, loss_fn, device)
    test_losses = calculate_loss(model, test_loader, loss_fn, device)
    forget_losses = calculate_loss(model, forget_loader, loss_fn, device)
    
    # Prepare the dataset for the classifier
    X_train = np.concatenate((train_losses, test_losses)).reshape(-1, 1)
    y_train = np.concatenate((np.ones(len(train_losses)), \
                              np.zeros(len(test_losses))))
    
    # Train the classifier
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    
    # Apply classifier to forget set
    forget_predictions = classifier.predict(forget_losses.reshape(-1, 1))
    
    # Evaluate classifier on forget set
    tn, fp, fn, tp = confusion_matrix(np.zeros(len(forget_losses)), \
                                      forget_predictions, labels=[0, 1]).ravel()
    
    # Calculate the ratio of true negatives to the size of the forget set
    true_negatives_ratio = tn / len(forget_losses)
    
    return true_negatives_ratio