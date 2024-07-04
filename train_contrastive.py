import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import yaml
from datetime import datetime
from models import ComparisonModel, get_model
from data import get_cifar100_dataloaders
import copy

def calculate_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
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
            'train_student_loss': 0.0,
            'train_classifier_loss': 0.0,
            'train_performance_loss': 0.0,
            'train_top1': 0.0,
            'train_top5': 0.0,
            'val_loss': 0.0,
            'val_student_loss': 0.0,
            'val_classifier_loss': 0.0,
            'val_performance_loss': 0.0,
            'val_top1': 0.0,
            'val_top5': 0.0
        }
    else:
        raise ValueError(f'Unknown mode: {mode}')

def update_metrics(metrics, loss_student, loss_classifier, loss_performance, top1, top5, mode='train'):
    metrics[f'{mode}_student_loss'] += loss_student.item()
    metrics[f'{mode}_classifier_loss'] += loss_classifier.item()
    metrics[f'{mode}_performance_loss'] += loss_performance.item()
    metrics[f'{mode}_top1'] += top1.item()
    metrics[f'{mode}_top5'] += top5.item()

def average_metrics(metrics, dataset_size, mode='train'):
    metrics[f'{mode}_student_loss'] /= dataset_size
    metrics[f'{mode}_classifier_loss'] /= dataset_size
    metrics[f'{mode}_performance_loss'] /= dataset_size
    metrics[f'{mode}_top1'] /= dataset_size
    metrics[f'{mode}_top5'] /= dataset_size

def log_metrics(writer, metrics, epoch, mode='train'):
    writer.add_scalars(f'Loss/{mode}', {'student_loss' : metrics[f'{mode}_student_loss'],
                                        'classifier_loss' : metrics[f'{mode}_classifier_loss']}, epoch)
    writer.add_scalar(f'Accuracy/{mode}_top1', metrics[f'{mode}_top1'], epoch)
    writer.add_scalar(f'Accuracy/{mode}_top5', metrics[f'{mode}_top5'], epoch)

def save_config(config, filename):
    with open(filename, 'w') as f:
        yaml.dump(config, f)

def load_config(filename):
    with open(filename, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def get_current_datetime_string():
    """
    Returns the current date and time as a formatted string.
    Format: 'MM-DD-HH:MM:SS'
    """
    now = datetime.now()
    return now.strftime("%m-%d-%H:%M:%S")

def train_models(student, teacher, classifier, train_loader, val_loader, num_epochs=10, learning_rate=0.001,
                 log_dir='runs', device='cuda', name_prefix=get_current_datetime_string()):
    # Define loss function and optimizers
    criterion = nn.BCELoss()
    val_criterion = nn.CrossEntropyLoss()
    student_optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(os.path.join(log_dir, name_prefix))

    for epoch in range(num_epochs):
        student.train()
        classifier.train()
        metrics = initialize_metrics(mode='train')

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = inputs.size(0)

            # Get outputs from student and teacher models
            student_outputs = student(inputs)
            teacher_outputs = teacher(inputs)

            # Create labels for classifier
            same_labels = torch.ones(batch_size, 1).to(device)
            different_labels = torch.zeros(batch_size, 1).to(device)

            # Shuffle inputs to create different pairs
            shuffled_indices = torch.randperm(batch_size)
            shuffled_teacher_outputs = teacher_outputs[shuffled_indices]

            # Combine same and different pairs
            combined_student_outputs = torch.cat((student_outputs, student_outputs), dim=0)
            combined_teacher_outputs = torch.cat((teacher_outputs, shuffled_teacher_outputs), dim=0)
            combined_labels = torch.cat((same_labels, different_labels), dim=0)

            # Train classifier
            classifier_optimizer.zero_grad()
            classifier_pred = classifier(combined_student_outputs.detach(), combined_teacher_outputs.detach())
            classifier_loss_value = criterion(classifier_pred, combined_labels)
            classifier_loss_value.backward()
            classifier_optimizer.step()

            # Train student to mimic teacher
            student_optimizer.zero_grad()
            student_pred = classifier(student_outputs, teacher_outputs)
            student_loss_value = criterion(student_pred, same_labels)
            student_loss_value.backward()
            student_optimizer.step()
            with torch.no_grad():
                pred = student(inputs)
                performance_loss = val_criterion(pred, labels)
                student_top1, student_top5 = calculate_accuracy(pred, labels, topk=(1, 5))
                update_metrics(metrics, student_loss_value, classifier_loss_value, performance_loss, student_top1, student_top5, mode='train')

        average_metrics(metrics, len(train_loader), mode='train')

        # Validation loop
        student.eval()
        classifier.eval()

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                batch_size = inputs.size(0)

                # Get outputs from student and teacher models
                student_outputs = student(inputs)
                teacher_outputs = teacher(inputs)

                # Create labels for classifier
                same_labels = torch.ones(batch_size, 1).to(device)
                different_labels = torch.zeros(batch_size, 1).to(device)

                # Shuffle inputs to create different pairs
                shuffled_indices = torch.randperm(batch_size)
                shuffled_teacher_outputs = teacher_outputs[shuffled_indices]

                # Combine same and different pairs
                combined_student_outputs = torch.cat((student_outputs, student_outputs), dim=0)
                combined_teacher_outputs = torch.cat((teacher_outputs, shuffled_teacher_outputs), dim=0)
                combined_labels = torch.cat((same_labels, different_labels), dim=0)

                # Validate classifier
                classifier_pred = classifier(combined_student_outputs, combined_teacher_outputs)
                classifier_loss_value = criterion(classifier_pred, combined_labels)

                # Validate student
                student_pred = classifier(student_outputs, teacher_outputs)
                student_loss_value = criterion(student_pred, same_labels)
                pred = student(inputs)
                performance_loss = val_criterion(pred, labels)
                student_top1, student_top5 = calculate_accuracy(pred, labels, topk=(1, 5))
                update_metrics(metrics, student_loss_value, classifier_loss_value, performance_loss, student_top1,
                               student_top5, mode='val')

        average_metrics(metrics, len(val_loader), mode='val')

        # Log training and validation metrics to TensorBoard
        log_metrics(writer, metrics, epoch, mode='train')
        log_metrics(writer, metrics, epoch, mode='val')

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {metrics["train_student_loss"]:.4f}, Train Top-1 Accuracy: {metrics["train_top1"]:.2f}%, Train Top-5 Accuracy: {metrics["train_top5"]:.2f}%')
        print(f'                  Val Loss: {metrics["val_student_loss"]:.4f}, Val Top-1 Accuracy: {metrics["val_top1"]:.2f}%, Val Top-5 Accuracy: {metrics["val_top5"]:.2f}%')

    # Save the models
    student_save_path = os.path.join(log_dir, f'{name_prefix}_student_model.pth')
    torch.save(student.state_dict(), student_save_path)
    print(f'Student model saved to {student_save_path}')
    classifier_save_path = os.path.join(log_dir, f'{name_prefix}_classifier_model.pth')
    torch.save(classifier.state_dict(), classifier_save_path)
    print(f'Classifier model saved to {classifier_save_path}')
    writer.close()

def main(args):
    # Generate example data
    train_loader, val_loader, test_loader, retain_loader, forget_loader = get_cifar100_dataloaders(
        batch_size=args.batch_size, validation_split=0.1,
        num_workers=2, random_seed=42,
        data_dir=args.data_dir)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    classifier = ComparisonModel(input_dim=100, feature_size=64, temperature=0.1, use_linear=False, use_norm=False).to(device)
    # Train the models
    model = get_model(args.model, num_classes=100, pretrained_weights=None,
                      weight_path=args.weight_path)
    teacher = copy.deepcopy(model)
    for param in teacher.parameters():
        param.requires_grad = False
    student =  get_model(args.model, num_classes=100, pretrained_weights=None, weight_path=args.weight_path_student)
    classifier.to(device)
    student.to(device)
    teacher.to(device)
    train_models(student, teacher, classifier, train_loader, val_loader, num_epochs=args.num_epochs,
                 learning_rate=args.learning_rate, log_dir=args.log_dir, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a student model to mimic a teacher model with a classifier.")
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train for.')
    parser.add_argument('--data_dir', type=str, default='./data/cifar100', help='Directory to save dataset.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory to save logs.')
    parser.add_argument('--config_dir', type=str, default='configs', help='Directory to save config.')
    parser.add_argument('--config_file', type=str, help='Path to configuration file to load.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training (e.g., "cpu", "cuda").')
    parser.add_argument('--name_prefix', type=str, default=None,
                        help='Define name prefix to store results (same prefix is used for logs, checkpoints, weights, etc).')
    parser.add_argument('--weight_path', type=str, help='Path to model weights file.')
    parser.add_argument('--weight_path_student', type=str, default=None, help='Path to model weights file to initialize student model.')
    parser.add_argument('--model', type=str, default='resnet18', help='Model architecture to use.')

    args = parser.parse_args()
    if args.config_file:
        config = load_config(os.path.join(args.config_dir, args.config_file))
        args = argparse.Namespace(**config)
    else:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.config_dir, exist_ok=True)
        config = vars(args)
        save_config(config, os.path.join(args.config_dir, get_current_datetime_string() + '.yaml'))
    main(args)