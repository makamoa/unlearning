from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import yaml

def get_current_datetime_string():
    """
    Returns the current date and time as a formatted string.
    Format: 'MM-DD-HH:MM:SS'
    """
    now = datetime.now()
    return now.strftime("%m-%d-%H:%M:%S")

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
    
def initialize_additional_metrics(keys, mode=''):
    return {f'{mode}{key}': 0.0 for key in keys}

def update_metrics(metrics, loss, top1, top5, mode='train'):
    metrics[f'{mode}_loss'] += loss.item()
    metrics[f'{mode}_top1'] += top1.item()
    metrics[f'{mode}_top5'] += top5.item()

def update_additional_metrics(metrics, mode='', **kwargs):
    for key, value in kwargs.items():
        if isinstance(value, (int, float)):
            metrics[f'{mode}{key}'] += value
        elif hasattr(value, 'item'):
            metrics[f'{mode}{key}'] += value.item()
        else:
            raise ValueError(f"Unsupported type for metric '{key}': {type(value)}")

def average_metrics(metrics, dataset_size, mode='train'):
    metrics[f'{mode}_loss'] /= dataset_size
    metrics[f'{mode}_top1'] /= dataset_size
    metrics[f'{mode}_top5'] /= dataset_size

def average_additional_metrics(metrics, dataset_size, keys, mode=''):
    for key in keys:
        metric_key = f'{mode}{key}'
        if metric_key in metrics:
            metrics[metric_key] /= dataset_size
        else:
            raise KeyError(f"Metric '{metric_key}' not found in metrics dictionary.")

def log_metrics(writer, metrics, epoch, mode='train'):
    writer.add_scalar(f'Loss/{mode}', metrics[f'{mode}_loss'], epoch)
    writer.add_scalar(f'Accuracy/{mode}_top1', metrics[f'{mode}_top1'], epoch)
    writer.add_scalar(f'Accuracy/{mode}_top5', metrics[f'{mode}_top5'], epoch)

def log_additional_metrics(writer, metrics, epoch, mode, keys):
    for key in keys:
        writer.add_scalar(f'{key}/{mode}', metrics[f'{mode}{key}'], epoch)
        print('log_added', f'{key}/{mode}', metrics[f'{mode}{key}'], epoch)

def log_metrics_unlearn(writer, metrics_retain, metrics_forget, epoch, mode='train'):
    writer.add_scalars(f'Loss/{mode}', {'retain': metrics_retain[f'{mode}_loss']}, epoch)
    writer.add_scalars(f'Accuracy/{mode}_top1',
                       {'retain': metrics_retain[f'{mode}_top1'], 'forget': metrics_forget[f'{mode}_top1']}, epoch)
    writer.add_scalars(f'Accuracy/{mode}_top5',
                       {'retain': metrics_retain[f'{mode}_top5'], 'forget': metrics_forget[f'{mode}_top5']}, epoch)
    

def print_metrics(metrics, epoch, num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}, '
          f'Train Loss: {metrics["train_loss"]:.4f}, '
          f'Train Top-1 Accuracy: {metrics["train_top1"]:.2f}%, '
          f'Train Top-5 Accuracy: {metrics["train_top5"]:.2f}%, '
          f'Validation Loss: {metrics["val_loss"]:.4f}, '
          f'Validation Top-1 Accuracy: {metrics["val_top1"]:.2f}%, '
          f'Validation Top-5 Accuracy: {metrics["val_top5"]:.2f}%')
    
def print_additional_metrics(metrics, epoch, num_epochs, end='\n'):
    print(f'Epoch {epoch + 1}/{num_epochs}', end=' ')
    for key, value in metrics.items():
        print(f'{key} : {value}', end=' ')
    print(end)

def log_metrics_unlearn_additional(writer, metrics_objective, metrics_condition, metrics_validation, epoch, stopping_criterion='unlearning'):
    if stopping_criterion in ['unlearning', None]:
        writer.add_scalars(f'Loss', {'objective' : metrics_objective['loss'],
                                     'condition' : metrics_condition['loss'],
                                     'validation': metrics_validation['loss']}, epoch)
    elif stopping_criterion in ['refining', 'confusion']:
        writer.add_scalars(f'Loss', {'objective' : metrics_objective['loss'], 
                                     'condition' : metrics_condition['loss'],
                                     'validation': metrics_validation['loss'],
                                     'loss_forget' : metrics_condition['loss_forget']}, epoch)
    writer.add_scalars(f'Accuracy/top1', {'objective' : metrics_objective['top1'], 'condition' : metrics_condition['top1'], 'validation': metrics_validation['top1']}, epoch)
    writer.add_scalars(f'Accuracy/top5', {'objective' : metrics_objective['top5'], 'condition' : metrics_condition['top5'], 'validation': metrics_validation['top5']}, epoch)

def build_name_prefix(args):
    prefix = ''
    if not args.untrain:
        prefix = f'model_{args.model}_lr_{args.learning_rate}_' + get_current_datetime_string()
    else:
        # old_prefix = os.path.basename(args.weight_path)[:len('_model.pth')]
        # prefix = 'untrained_' + old_prefix
        prefix = f'model_{args.model}_lr_{args.learning_rate}_untrain' \
            + get_current_datetime_string()
    return prefix

def save_config(config, filename):
    with open(filename, 'w') as f:
        yaml.dump(config, f)

def load_config(filename):
    with open(filename, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)