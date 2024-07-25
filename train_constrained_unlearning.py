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
from helper_functions import get_current_datetime_string, initialize_metrics, \
    update_metrics, update_additional_metrics, average_metrics, \
    average_additional_metrics, log_metrics, log_metrics_unlearn, \
    print_metrics, build_name_prefix, initialize_additional_metrics, \
    log_additional_metrics

def _grad_norm(model):
    total_norm = 0.0

    for param in model.parameters():
        if param.grad is not None:
            param_norm = torch.norm(param.grad, 2.)
            total_norm += param_norm.item() ** 2.

    total_norm = total_norm ** .5
    return total_norm    

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

def untrain_constrained(model,
                        retainloader,
                        forgetloader,
                        unseenloader,
                        validloader,
                        num_epochs=10,
                        learning_rate=0.001,
                        log_dir='runs',
                        device='cuda',
                        name_prefix=get_current_datetime_string()) -> None:
    """
    Unlearn a model based on the constrained problem formulation.

    Args:
        model: The neural network model that requires unlearning.
        retainloader: DataLoader for the data to be retained.
        forgetloader: DataLoader for the data to be forgotten.
        unseenloader: Dataloader for the unseen data used implicitly in
            training.
        validloader: DataLoader for the validation data.
        num_epochs: Number of epochs to untrain.
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
    base_optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # Define the teacher model
    model_teacher = copy.deepcopy(model)
    for param in model_teacher.parameters():
        param.requires_grad = False

    epsilon = torch.tensor(0.).requires_grad_(False)
    alpha = 10.

    # Run unlearning epochs
    for epoch in range(num_epochs):
        model.train()
        metrics_retain = initialize_metrics()
        metrics_forget = initialize_metrics()
        metrics_both = initialize_additional_metrics(['grad'])
        n_batches = 0

        # Untraining loop
        assert len(forgetloader) > 0
        assert len(forgetloader) <= len(retainloader)

        print(f'Current alpha = {alpha}')
        for retain_batch, forget_batch \
                in zip(retainloader, forgetloader):
            n_batches += 1
            retain_inputs, retain_labels = retain_batch
            forget_inputs, forget_labels = forget_batch
            retain_inputs, retain_labels = retain_inputs.to(device), retain_labels.to(device)
            forget_inputs, forget_labels = forget_inputs.to(device), forget_labels.to(device)

            forget_loss = - criterion(model(forget_inputs), forget_labels)
            retain_loss = criterion(model(retain_inputs), retain_labels)

            if epoch == 0:
                epsilon += retain_loss.clone().detach().cpu()
                base_optimizer.zero_grad()
                continue

            # penalty = -1 * alpha * torch.log(epsilon - retain_loss) # logarithmic barrier; does not work with SGD
            penalty = torch.square(torch.max(torch.tensor(0), retain_loss - epsilon)) # quadratic penalty
            # penalty = torch.pow(torch.max(torch.tensor(0), retain_loss - epsilon), 4) # quartic penalty
            # penalty = torch.exp(torch.max(torch.tensor(0), retain_loss - epsilon)) - 1.
            loss = forget_loss + alpha * penalty

            base_optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            curr_grad_norm = _grad_norm(model)
            base_optimizer.step()

            top1, top5 = calculate_accuracy(model(retain_inputs),
                                            retain_labels, topk=(1, 5))
            update_metrics(metrics_retain, retain_loss, top1, top5,
                           mode='train')
            top1, top5 = calculate_accuracy(model(forget_inputs),
                                            forget_labels, topk=(1, 5))
            update_metrics(metrics_forget, -forget_loss, top1, top5,
                           mode='train')
            update_additional_metrics(metrics_both, grad=curr_grad_norm)

        average_metrics(metrics_retain, n_batches, mode='train')
        average_metrics(metrics_forget, n_batches, mode='train')
        average_additional_metrics(metrics_both, n_batches, ['grad'])
        if epoch == 0:
            epsilon /= n_batches
            print(f'epsilon = {epsilon}')
        if metrics_both['train_grad'] < 5. and epoch != 0:
            alpha *= 1.05
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
        log_additional_metrics(writer, metrics_both, epoch, 'train', ['grad'])

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


def save_config(config, filename):
    with open(filename, 'w') as f:
        yaml.dump(config, f)


def load_config(filename):
    with open(filename, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def main(args):
    # Define CIFAR100 dataset handler
    dataset_handler = data.CIFAR100Handler(batch_size=args.batch_size,
                                           validation_split=0.1,
                                           random_seed=42,
                                           data_dir=args.data_dir)
    data_confuser = data.uniform_confuser(confuse_level=.0, random_seed=42)
    splitter = data.mix_both_sets(amend_split=1., retain_split=0.1, random_seed=42)
    confused_dataset_handler = data.AmendedDatasetHandler(dataset_handler, data_confuser, splitter)
    train_loader, val_loader, test_loader, forget_loader, retain_loader, unseen_loader = \
        confused_dataset_handler.get_dataloaders()
    # train_loader, val_loader, test_loader, retain_loader, forget_loader = data.get_cifar100_dataloaders(batch_size=args.batch_size, validation_split=0.1,
    #                                                                  num_workers=2, random_seed=42,
    #                                                                  data_dir=args.data_dir)

    # forget_loader, train_loader = data.CorrespondingLoaders(forget_loader, train_loader).get_loaders() # sync labels for train and forget
    # Initialize model
    model = get_model(args.model, num_classes=100, pretrained_weights=None,
                      weight_path=args.weight_path)
    device = torch.device(args.device if torch.cuda.is_available() or \
                                         'cpu' not in args.device else 'cpu')
    model.to(device)
    # Untrain the model
    untrain_constrained(model,
                        retain_loader,
                        forget_loader,
                        unseen_loader,
                        val_loader,
                        num_epochs=args.untrain_num_epochs,
                        learning_rate=args.learning_rate,
                        log_dir=args.log_dir,
                        device=args.device,
                        name_prefix=args.name_prefix)


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
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training (e.g., "cpu", "cuda", "cuda:0", "cuda:1").')
    parser.add_argument('--rho', type=float, default=None, help='SAM radius parameter')
    parser.add_argument('--name_prefix', type=str, default=None,
                        help='Define name prefix to store results (same prefix is used for logs, checkpoints, weights, etc).')
    parser.add_argument('--untrain', type=bool, default=False, help='SAM radius parameter')
    parser.add_argument('--sam_lr', type=float, default=0.1, help='Learning rate for the SAM base optimizer')
    parser.add_argument('--kl_retain_lr', type=float, default=0.1,
                        help='Learning rate for the remaining part of the retain loss')
    parser.add_argument('--kl_forget_lr', type=float, default=0.1, help='Learning rate for the forget loss')
    parser.add_argument('--untrain_num_epochs', type=int, default=5, help='Number of epochs to untrain for.')
    parser.add_argument('--SCRUB', type=bool, default=False, help='Use SCRUB optimizer or not for untraining')

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