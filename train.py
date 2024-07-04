import argparse
import yaml
import os
import copy
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from datetime import datetime
from models import get_model
import data
from data import get_cifar100_dataloaders, CorrespondingLoaders
from optimizer import SAM, base_loss, KL_retain_loss, KL_forget_loss, inverse_KL_forget_loss
import torch.optim.lr_scheduler as lr_scheduler

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


def initialize_metrics(mode='train'):
    if mode == 'train':
        return {
            'train_loss': 0.0,
            'train_top1': 0.0,
            'train_top5': 0.0,
            'val_loss': 0.0,
            'val_top1': 0.0,
            'val_top5': 0.0
        }
    else:
        raise ValueError(f'Unknown mode: {mode}')

def update_metrics(metrics, loss, top1, top5, mode='train'):
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

def log_metrics_unlearn(writer, metrics_retain, metrics_forget, epoch, mode='train'):
    writer.add_scalars(f'Loss/{mode}', {'retain' : metrics_retain[f'{mode}_loss']}, epoch)
    writer.add_scalars(f'Accuracy/{mode}_top1', {'retain' : metrics_retain[f'{mode}_top1'], 'forget' : metrics_forget[f'{mode}_top1']}, epoch)
    writer.add_scalars(f'Accuracy/{mode}_top5', {'retain' : metrics_retain[f'{mode}_top5'], 'forget' : metrics_forget[f'{mode}_top5']}, epoch)

def print_metrics(metrics, epoch, num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {metrics["train_loss"]:.4f}, '
          f'Train Top-1 Accuracy: {metrics["train_top1"]:.2f}%, '
          f'Train Top-5 Accuracy: {metrics["train_top5"]:.2f}%, '
          f'Validation Loss: {metrics["val_loss"]:.4f}, '
          f'Validation Top-1 Accuracy: {metrics["val_top1"]:.2f}%, '
          f'Validation Top-5 Accuracy: {metrics["val_top5"]:.2f}%')

def build_name_prefix(args):
    prefix = ''
    if not args.untrain:
        prefix = f'model_{args.model}_sam_{args.use_sam}_rho_{args.rho}_lr_{args.learning_rate}_' + get_current_datetime_string()
    else:
        old_prefix = os.path.basename(args.weight_path)[:len('_model.pth')]
        prefix = 'untrained_' + old_prefix
    return prefix


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
            update_metrics(metrics, loss, top1, top5, mode='train')

        average_metrics(metrics, len(train_loader), mode='train')

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
                update_metrics(metrics, loss, top1, top5, mode='val')

        average_metrics(metrics, len(val_loader), mode='val')
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

def untrain_model(model, retainloader, forgetloader, validloader, num_epochs=10, learning_rate=0.001,
                  log_dir='runs', device='cuda', name_prefix=get_current_datetime_string(), use_sam=False, rho=0.05) -> None:
    """
    Unlearn a model based on the problem formulation
    `minimize retainloss + forgetloss`. 

    Args:
        model: The neural network model that requires unlearning.
        retainloader: DataLoader for the data to be retained.
        forgetloader: DataLoader for the data to be forgotten.
        validloader: DataLoader for the validation data.
        num_epochs: Number of epochs to untrain.
        retainloss_sam: Loss function for retained data, on which SAM
            (Sharpness-Aware Minimization) is applied. The total retention
            loss is the sum of retainloss_sam and retainloss_rest.
        retainloss_rest: Loss function for retained data, on which
            a standard optimizer is applied.
        forgetloss: Loss function for forgotten data.
        log_dir: Directory to save TensorBoard logs.
        device: Device to run the model on (e.g., 'cpu' or 'cuda').
    """

    if name_prefix is None:
        name_prefix = get_current_datetime_string()

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(os.path.join(log_dir, name_prefix))

    # Define loss function for validation
    criterion = nn.CrossEntropyLoss()
    # Define optimizers
    if use_sam:
        base_optimizer = SAM(model.parameters(), optim.SGD, lr=learning_rate, momentum=0.9, rho=rho)
    else:
        base_optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    KL_retain_optimizer = optim.SGD(model.parameters(), lr=args.kl_retain_lr, momentum=0.9)
    KL_forget_optimizer = optim.SGD(model.parameters(), lr=args.kl_forget_lr, momentum=0.9)
    # Define the teacher model
    model_teacher = copy.deepcopy(model)
    for param in model_teacher.parameters():
        param.requires_grad = False

    # Run unlearning epochs
    for epoch in range(num_epochs):
        model.train()
        metrics_retain = initialize_metrics()
        metrics_forget = initialize_metrics()
        n_batches = 0

        # Untraining loop
        assert len(forgetloader) > 0
        assert len(forgetloader) <= len(retainloader)
        for retain_batch, forget_batch \
                in zip(retainloader, forgetloader):
            n_batches += 1
            retain_inputs, retain_labels = retain_batch
            forget_inputs, forget_labels = forget_batch
            retain_inputs, retain_labels = retain_inputs.to(device), retain_labels.to(device)
            forget_inputs, forget_labels = forget_inputs.to(device), forget_labels.to(device)
            # sam_retain_optimizer stage
            retain_loss_value = perform_optimizer_step(model,
                                                        model_teacher,
                                                        retain_inputs,
                                                        retain_labels,
                                                        base_loss,
                                                        base_optimizer,
                                                        use_sam=use_sam
                                                        )
            perform_optimizer_step(model,
                                    model_teacher,
                                    retain_inputs,
                                    retain_labels,
                                    KL_retain_loss,
                                    KL_retain_optimizer,
                                    False
                                    )
            # forget_optimizer_stage
            perform_optimizer_step(model, model_teacher, forget_inputs,
                                   forget_labels, inverse_KL_forget_loss,
                                   KL_forget_optimizer, False)
            
            top1, top5 = calculate_accuracy(model(retain_inputs),
                                            retain_labels, topk=(1, 5))
            update_metrics(metrics_retain, retain_loss_value, top1, top5,
                           mode='train')
            top1, top5 = calculate_accuracy(model(forget_inputs),
                                            forget_labels, topk=(1, 5))
            update_metrics(metrics_forget, retain_loss_value, top1, top5,
                           mode='train')

        average_metrics(metrics_retain, n_batches, mode='train')
        average_metrics(metrics_forget, n_batches, mode='train')
        # Validation loop
        model.eval()
        with torch.no_grad():
            for inputs, labels in validloader:
                # Move data to the appropriate device
                inputs, labels = inputs.to(device), labels.to(device)
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                top1, top5 = calculate_accuracy(outputs, labels, topk=(1, 5))
                update_metrics(metrics_retain, loss, top1, top5, mode='val')

        average_metrics(metrics_retain, len(validloader), mode='val')
        print('Retain Set Performance')
        print_metrics(metrics_retain, epoch, num_epochs)
        print('Forget Set Performance')
        print_metrics(metrics_forget, epoch, num_epochs)
        log_metrics_unlearn(writer, metrics_retain, metrics_forget, epoch, mode='train')
        log_metrics(writer, metrics_retain, epoch, mode='val')

    # Save the final model
    model_save_path = os.path.join(log_dir, f"{name_prefix}_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

    # Save model graph
    sample_inputs = next(iter(retainloader))[0].to(device)
    writer.add_graph(model, sample_inputs)

    # Close the TensorBoard writer
    writer.close()
    print('Unlearning complete')

def perform_optimizer_step(model, model_teacher, inputs, labels, loss_fn, \
                           optimizer, use_sam):
    """
    Perform a single optimization step.

    Args:
        model: The neural network model.
        model_teacher: The teacher neural network model.
        inputs: Input batch
        labels: Ground truth labels
        loss_fn: Loss function that accepts model, model_teacher,
            inputs, labels as parameters
        optimizer: Optimizer.
        use_sam: Boolean to indicate whether to use SAM optimizer.

    Returns:
        loss_value: Calculated loss value.
    """
    if use_sam:
        loss_value = loss_fn(model, model_teacher, inputs, labels)
        loss_value.backward()
        optimizer.first_step(zero_grad=True)

        loss_value = loss_fn(model, model_teacher, inputs, labels)
        loss_value.backward()
        optimizer.second_step(zero_grad=True)
    else:
        loss_value = loss_fn(model, model_teacher, inputs, labels)
        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss_value

def save_config(config, filename):
    with open(filename, 'w') as f:
        yaml.dump(config, f)


def load_config(filename):
    with open(filename, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def main(args):
    # Define CIFAR100 dataset handler
    # dataset_handler = data.CIFAR100Handler(batch_size=args.batch_size,
    #                                        validation_split=0.1,
    #                                        random_seed=42,
    #                                        data_dir=args.data_dir)
    # data_confuser = data.uniform_confuser(confuse_level=.0, random_seed=42)
    # splitter = data.mix_both_sets(amend_split=1., retain_split=0.1, random_seed=42)
    # confused_dataset_handler = data.AmendedDatasetHandler(dataset_handler, data_confuser, splitter, class_wise_corr=True)
    # train_loader, val_loader, test_loader, forget_loader, retain_loader, unseen_loader = \
    #     confused_dataset_handler.get_dataloaders()
    train_loader, val_loader, test_loader, retain_loader, forget_loader = get_cifar100_dataloaders(batch_size=args.batch_size, validation_split=0.1,
                                                                     num_workers=2, random_seed=42,
                                                                     data_dir=args.data_dir)
    forget_loader, val_loader = CorrespondingLoaders(forget_loader, val_loader).get_loaders()
    # Initialize model
    model = get_model(args.model, num_classes=100, pretrained_weights=None,
                      weight_path=args.weight_path)
    device = torch.device(args.device if torch.cuda.is_available() or \
                          'cpu' not in args.device else 'cpu')
    model.to(device)

    # Train the model
    if not args.untrain:
        train_model(model, train_loader, val_loader, num_epochs=args.num_epochs,
                    learning_rate=args.learning_rate, log_dir=args.log_dir,
                    device=device, use_sam=args.use_sam, rho=args.rho,
                    name_prefix=args.name_prefix)
    else:
        untrain_model(model, retain_loader, forget_loader, val_loader, num_epochs=args.untrain_num_epochs,
                      log_dir=args.log_dir, device=args.device, name_prefix=args.name_prefix, use_sam=args.use_sam)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on CIFAR-100 with TensorBoard logging.")
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--data_dir', type=str, default='./data/cifar100', help='Directory to save dataset.')
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory to save logs.')
    parser.add_argument('--config_dir', type=str, default='runs', help='Directory to save config.')
    parser.add_argument('--config_file', type=str, help='Path to configuration file to load.')
    parser.add_argument('--model', type=str, default='resnet18', help='Model architecture to use.')
    parser.add_argument('--weight_path', type=str, help='Path to model weights file.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., "cpu", "cuda", "cuda:0", "cuda:1").')
    parser.add_argument('--use_sam', action='store_true', help='Whether to use SAM optimizer or not')
    parser.add_argument('--rho', type=float, default=None, help='SAM radius parameter')
    parser.add_argument('--name_prefix', type=str, default=None,
                        help='Define name prefix to store results (same prefix is used for logs, checkpoints, weights, etc).')
    parser.add_argument('--untrain', type=bool, default=False, help='SAM radius parameter')
    parser.add_argument('--sam_lr', type=float, default=0.1, help='Learning rate for the SAM base optimizer')
    parser.add_argument('--kl_retain_lr', type=float, default=0.1, help='Learning rate for the remaining part of the retain loss')
    parser.add_argument('--kl_forget_lr', type=float, default=0.1, help='Learning rate for the forget loss')
    parser.add_argument('--untrain_num_epochs', type=int, default=5, help='Number of epochs to untrain for.')

    args = parser.parse_args()
    if args.untrain:
        assert args.weight_path is not None
    if args.name_prefix is None:
        args.name_prefix = build_name_prefix(args)
    if args.config_file:
        config = load_config(os.path.join(args.config_dir, args.config_file))
        args = argparse.Namespace(**config)
    else:
        os.makedirs(args.log_dir, exist_ok=True)
        config = vars(args)
        save_config(config, os.path.join(args.config_dir, args.name_prefix + '.yaml'))
    main(args)
