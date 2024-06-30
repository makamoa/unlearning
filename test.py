import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import get_model
from data import get_cifar100_dataloaders
from models import membership_inference_attack


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


def test_model(model, test_loader, device='cuda'):
    """
    Tests the given model on the provided test data.

    :param model: The model to be tested.
    :param test_loader: DataLoader for the test data.
    :param device: Device to use for testing (e.g., 'cuda' or 'cpu').
    :return: Dictionary with test loss, top-1 accuracy, and top-5 accuracy.
    """

    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            top1, top5 = calculate_accuracy(outputs, labels, topk=(1, 5))
            correct_top1 += top1.item() * inputs.size(0)
            correct_top5 += top5.item() * inputs.size(0)
            total += labels.size(0)

    test_loss /= total
    accuracy_top1 = correct_top1 / total
    accuracy_top5 = correct_top5 / total

    return {
        'test_loss': test_loss,
        'test_top1_accuracy': accuracy_top1,
        'test_top5_accuracy': accuracy_top5
    }


def main(args):
    # Load CIFAR-100 dataset
    train_loader, _, test_loader = get_cifar100_dataloaders(batch_size=args.batch_size, validation_split=0.1, num_workers=2,
                                                 random_seed=42, data_dir=args.data_dir)

    # Initialize model
    model = get_model(args.model, num_classes=100, pretrained=False)
    device = torch.device(args.device if torch.cuda.is_available() or 'cpu' not in args.device else 'cpu')
    model.to(device)

    # Load the saved model weights
    model.load_state_dict(torch.load(args.weight_path))
    # Load the saved model weights
    dummy_model = get_model(args.model, num_classes=100, pretrained=False)
    device = torch.device(args.device if torch.cuda.is_available() or 'cpu' not in args.device else 'cpu')
    dummy_model.to(device)

    # Test the model
    test_metrics = test_model(model, test_loader, device)
    mia = membership_inference_attack(model, train_loader, test_loader)
    dummy_mia = membership_inference_attack(dummy_model, train_loader, test_loader)

    print(f"Test Loss: {test_metrics['test_loss']:.4f}")
    print(f"Test Top-1 Accuracy: {test_metrics['test_top1_accuracy']:.2f}%")
    print(f"Test Top-5 Accuracy: {test_metrics['test_top5_accuracy']:.2f}%")
    print(f"Membership Inference Between Train and Test Set: Accuracy - {mia['accuracy']*100:.2f}%; Precision - {mia['precision']*100:.2f}%; Recall - {mia['recall']*100:.2f}%; Balanced Accuracy - {mia['balanced_accuracy']*100:.2f}%;")
    print(
        f"Dummy Membership Inference Between Train and Test Set: Accuracy - {dummy_mia['accuracy'] * 100:.2f}%; Precision - {dummy_mia['precision'] * 100:.2f}%; Recall - {dummy_mia['recall'] * 100:.2f}%; Balanced Accuracy - {dummy_mia['balanced_accuracy']*100:.2f}%")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a model on CIFAR-100.")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing.')
    parser.add_argument('--data_dir', type=str, default='./data/cifar100', help='Directory to save dataset.')
    parser.add_argument('--model', type=str, default='resnet18', help='Model architecture to use.')
    parser.add_argument('--weight_path', type=str, default=True, help='Path to the model weights file.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for testing (e.g., "cpu", "cuda", "cuda:0", "cuda:1").')
    args = parser.parse_args()
    main(args)
