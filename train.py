import argparse
import yaml
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from datetime import datetime
from models import get_model
from data import get_cifar100_dataloaders
from optimizer import SAM

def get_current_datetime_string():
    """
    Returns the current date and time as a formatted string.
    Format: 'MM-DD-HH:MM:SS'
    """
    now = datetime.now()
    return now.strftime("%m-%d-%H:%M:%S")

def calculate_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def initialize_metrics():
    return {
        'train_loss': 0.0,
        'train_top1': 0.0,
        'train_top5': 0.0,
        'val_loss': 0.0,
        'val_top1': 0.0,
        'val_top5': 0.0
    }

def update_metrics(metrics, loss, top1, top5, inputs_size, mode='train'):
    metrics[f'{mode}_loss'] += loss.item()
    metrics[f'{mode}_top1'] += top1.item()
    metrics[f'{mode}_top5'] += top5.item()

def average_metrics(metrics, dataset_size, mode='train'):
    metrics[f'{mode}_loss'] /= dataset_size
    metrics[f'{mode}_top1'] /= dataset_size
    metrics[f'{mode}_top5'] /= dataset_size

def log_metrics(writer, metrics, epoch, mode='train'):
    writer.add_scalar(f'Loss/{mode}', metrics[f'{mode}_loss'], epoch)
    writer.add_scalar(f'Accuracy/{mode}_top1', metrics[f'{mode}_top1'], epoch)
    writer.add_scalar(f'Accuracy/{mode}_top5', metrics[f'{mode}_top5'], epoch)

def print_metrics(metrics, epoch, num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {metrics["train_loss"]:.4f}, '
          f'Train Top-1 Accuracy: {metrics["train_top1"]:.2f}%, '
          f'Train Top-5 Accuracy: {metrics["train_top5"]:.2f}%, '
          f'Validation Loss: {metrics["val_loss"]:.4f}, '
          f'Validation Top-1 Accuracy: {metrics["val_top1"]:.2f}%, '
          f'Validation Top-5 Accuracy: {metrics["val_top5"]:.2f}%')

def build_name_prefix(args):
    return f'sam_{args.use_sam}_rho_{args.rho}_lr_{args.learning_rate}_' + get_current_datetime_string()

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, log_dir='runs', device='cuda', use_sam=False, rho=0.05, name_prefix=get_current_datetime_string()):
    """
    Trains the given model on the provided training data and evaluates it on the validation data.

    :param model: The model to be trained.
    :param train_loader: Training data loader.
    :param val_loader: Validation data loader.
    :param num_epochs: Number of epochs to train for.
    :param learning_rate: Learning rate for the optimizer.
    :param log_dir: Directory to save TensorBoard logs.
    :param device: Device to use for training (e.g., 'cuda', 'cuda:0', 'cuda:1', 'cpu').
    :param use_sam: Boolean to indicate whether to use SAM optimizer.
    :param rho: Hyperparameter for SAM optimizer.
    :param name_prefix: Prefix for naming the log directory and saved model.
    """

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    if use_sam:
        base_optimizer = optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, lr=learning_rate, momentum=0.9, rho=rho)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(os.path.join(log_dir, name_prefix))

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        metrics = initialize_metrics()

        for inputs, labels in train_loader:
            # Move data to the appropriate device
            inputs, labels = inputs.to(device), labels.to(device)

            if use_sam:
                # First forward-backward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.first_step(zero_grad=True)

                # Second forward-backward pass
                criterion(model(inputs), labels).backward()
                optimizer.second_step(zero_grad=True)
            else:
                # Standard forward-backward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            top1, top5 = calculate_accuracy(outputs, labels, topk=(1, 5))
            update_metrics(metrics, loss, top1, top5, inputs.size(0), mode='train')

        average_metrics(metrics, len(train_loader.dataset), mode='train')

        # Validation loop
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Move data to the appropriate device
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                top1, top5 = calculate_accuracy(outputs, labels, topk=(1, 5))
                update_metrics(metrics, loss, top1, top5, inputs.size(0), mode='val')

        average_metrics(metrics, len(val_loader.dataset), mode='val')
        print_metrics(metrics, epoch, num_epochs)
        log_metrics(writer, metrics, epoch, mode='train')
        log_metrics(writer, metrics, epoch, mode='val')

    # Save the final model
    model_save_path = os.path.join(log_dir, f"{name_prefix}_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

    # Save model graph
    sample_inputs = next(iter(train_loader))[0].to(device)
    writer.add_graph(model, sample_inputs)

    # Close the TensorBoard writer
    writer.close()
    print('Training complete')

def save_config(config, filename):
    with open(filename, 'w') as f:
        yaml.dump(config, f)

def load_config(filename):
    with open(filename, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def main(args):
    # Load CIFAR-100 dataset
    train_loader, val_loader, test_loader = get_cifar100_dataloaders(batch_size=args.batch_size, validation_split=0.1, num_workers=2, random_seed=42, data_dir=args.data_dir)

    # Initialize model
    model = get_model(args.model, num_classes=100, pretrained=False, weight_path=args.weight_path)
    device = torch.device(args.device if torch.cuda.is_available() or 'cpu' not in args.device else 'cpu')
    model.to(device)

    # Train the model
    train_model(model, train_loader, val_loader, num_epochs=args.num_epochs, learning_rate=args.learning_rate, log_dir=args.log_dir, device=device, use_sam=args.use_sam, rho=args.rho, name_prefix=args.name_prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on CIFAR-100 with TensorBoard logging.")
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--data_dir', type=str, default='./data/cifar100', help='Directory to save dataset.')
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory to save logs and config.')
    parser.add_argument('--config_file', type=str, help='Path to configuration file to load.')
    parser.add_argument('--model', type=str, default='resnet18', help='Model architecture to use.')
    parser.add_argument('--weight_path', type=str, help='Path to model weights file.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., "cpu", "cuda", "cuda:0", "cuda:1").')
    parser.add_argument('--use_sam', action='store_true', help='Whether to use SAM optimizer or not')
    parser.add_argument('--rho', type=float, default=None, help='SAM radius parameter')
    parser.add_argument('--name_prefix', type=str, default=None,
                        help='Define name prefix to store results (same prefix is used for logs, checkpoints, weights, etc).')

    args = parser.parse_args()
    if args.name_prefix is None:
        args.name_prefix = build_name_prefix(args)
    if args.config_file:
        config = load_config(args.config_file)
        args = argparse.Namespace(**config)
    else:
        os.makedirs(args.log_dir, exist_ok=True)
        config = vars(args)
        save_config(config, os.path.join(args.log_dir, args.name_prefix + '.yaml'))
    main(args)
