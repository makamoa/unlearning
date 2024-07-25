import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import KLDivLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import yaml
from datetime import datetime
from models import ComparisonModel, get_model, SimpleBinaryClassifier, negative_loss, LogisticRegression
from data import get_cifar100_dataloaders, CorrespondingLoaders
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


def calculate_binary_accuracy(outputs, targets, threshold=0.5):
    """
    Calculate the accuracy of binary classification.

    Args:
        outputs (torch.Tensor): The output probabilities from the model of shape (batch_size, 1).
        targets (torch.Tensor): The ground truth labels of shape (batch_size, 1).
        threshold (float): The threshold for deciding class labels from output probabilities.

    Returns:
        float: The accuracy of the predictions.
    """
    # Convert probabilities to binary predictions using the threshold
    predicted_labels = (outputs > threshold).float()

    # Calculate the number of correct predictions
    correct_predictions = (predicted_labels == targets).float().sum()

    # Calculate the accuracy
    accuracy = correct_predictions / targets.size(0)

    return accuracy

def initialize_metrics(mode='train'):
    if mode == 'train':
        return {
            'train_student_loss': 0.0,
            'train_classifier_loss': 0.0,
            'train_performance_loss': 0.0,
            'train_top1': 0.0,
            'train_top5': 0.0,
            'train_top1_forget': 0.0,
            'train_top5_forget': 0.0,
            'train_binary_accuracy': 0.0,
            'val_loss': 0.0,
            'val_student_loss': 0.0,
            'val_classifier_loss': 0.0,
            'val_performance_loss': 0.0,
            'val_top1': 0.0,
            'val_top5': 0.0,
            'val_top1_forget': 0.0,
            'val_top5_forget': 0.0,
            'val_binary_accuracy': 0.0,
        }
    else:
        raise ValueError(f'Unknown mode: {mode}')

def update_metrics(metrics, loss_student, loss_classifier, loss_performance, top1, top5, top1_forget, top5_forget, binary_accuracy, mode='train'):
    metrics[f'{mode}_student_loss'] += loss_student.item()
    metrics[f'{mode}_classifier_loss'] += loss_classifier.item()
    metrics[f'{mode}_performance_loss'] += loss_performance.item()
    metrics[f'{mode}_top1'] += top1.item()
    metrics[f'{mode}_top5'] += top5.item()
    metrics[f'{mode}_top1_forget'] += top1_forget.item()
    metrics[f'{mode}_top5_forget'] += top5_forget.item()
    metrics[f'{mode}_binary_accuracy'] += binary_accuracy.item()

def average_metrics(metrics, dataset_size, mode='train'):
    metrics[f'{mode}_student_loss'] /= dataset_size
    metrics[f'{mode}_classifier_loss'] /= dataset_size
    metrics[f'{mode}_performance_loss'] /= dataset_size
    metrics[f'{mode}_top1'] /= dataset_size
    metrics[f'{mode}_top5'] /= dataset_size
    metrics[f'{mode}_top1_forget'] /= dataset_size
    metrics[f'{mode}_top5_forget'] /= dataset_size
    metrics[f'{mode}_binary_accuracy'] /= dataset_size

def log_metrics(writer, metrics, epoch, mode='train'):
    writer.add_scalars(f'Loss/{mode}', {'student_loss' : metrics[f'{mode}_student_loss'],
                                        'classifier_loss' : metrics[f'{mode}_classifier_loss']}, epoch)
    writer.add_scalars(f'Accuracy/{mode}_top1', {'retain' : metrics[f'{mode}_top1'], 'forget' : metrics[f'{mode}_top1_forget']}, epoch)
    writer.add_scalars(f'Accuracy/{mode}_top5', {'retain' : metrics[f'{mode}_top5'], 'forget' : metrics[f'{mode}_top5_forget']}, epoch)
    writer.add_scalar(f'Accuracy/{mode}_binary_accuracy', metrics[f'{mode}_binary_accuracy'], epoch)

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

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

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

def untrain_models_simple(student, teacher, classifier, train_loader, val_loader, retain_loader, forget_loader, num_epochs=10, learning_rate=0.001,
                 log_dir='runs', device='cuda', name_prefix=get_current_datetime_string(),
                   freeze_classifier=False, freeze_student=False,
                   stoppage=True):
    # Define loss function and optimizers
    criterion = nn.BCELoss()
    negative_criterion = negative_loss(criterion)
    val_criterion = nn.CrossEntropyLoss()
    student_optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(os.path.join(log_dir, name_prefix))
    forget_loader_ordered, val_loader_ordered = CorrespondingLoaders(forget_loader, val_loader, seed=42).get_loaders()
    for epoch in range(num_epochs):
        student.train()
        classifier.train()
        metrics = initialize_metrics(mode='train')

        for (inputs_retain, labels_retain), (inputs_forget, labels_forget), (inputs_val, labels_val) in zip(retain_loader, forget_loader_ordered, val_loader_ordered):
            inputs_retain = inputs_retain.to(device)
            labels_retain = labels_retain.to(device)
            inputs_forget = inputs_forget.to(device)
            labels_forget = labels_forget.to(device)
            inputs_val = inputs_val.to(device)
            labels_val = labels_val.to(device)
            batch_size = inputs_forget.size(0)

            # Get outputs from student and teacher models for forget
            student_outputs_forget = student(inputs_forget)
            teacher_outputs_forget = teacher(inputs_forget)
            # Get outputs from student and teacher models for val
            student_outputs_val = student(inputs_val)
            teacher_outputs_val = teacher(inputs_val)

            # Create labels for classifier
            forget_labels = torch.ones(batch_size, 1).to(device)
            val_labels = torch.zeros(batch_size, 1).to(device)

            # Shuffle inputs to create different pairs
            shuffled_indices = torch.randperm(2*batch_size)

            # Combine same and different pairs
            combined_outputs = torch.cat((student_outputs_forget, teacher_outputs_val), dim=0)
            combined_labels = torch.cat((forget_labels, val_labels), dim=0)
            # Shuffle forget and validation labels
            combined_outputs = combined_outputs[shuffled_indices]
            combined_labels = combined_labels[shuffled_indices]
            # Train classifier
            if not freeze_classifier:
                classifier_optimizer.zero_grad()
                classifier_pred = classifier(combined_outputs.detach())
                classifier_loss_value = criterion(classifier_pred, combined_labels)
                classifier_loss_value.backward()
                classifier_optimizer.step()

            # Train student to mimic teacher
            if not freeze_student:
                student_optimizer.zero_grad()
                student_pred = classifier(combined_outputs)
                student_loss_value = negative_criterion(student_pred, combined_labels)
                student_loss_value.backward()
                student_optimizer.step()

            with torch.no_grad():
                pred = student(inputs_retain)
                performance_loss = val_criterion(pred, labels_retain)
                student_top1, student_top5 = calculate_accuracy(pred, labels_retain, topk=(1, 5))
                classifier_pred = classifier(combined_outputs)
                binary_accuracy = calculate_binary_accuracy(classifier_pred, combined_labels)
                classifier_loss_value = criterion(classifier_pred, combined_labels)
                student_loss_value = negative_criterion(classifier_pred, combined_labels)
                update_metrics(metrics, -student_loss_value, classifier_loss_value, performance_loss, student_top1, student_top5, binary_accuracy, mode='train')

        average_metrics(metrics, len(forget_loader), mode='train')

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

                performance_loss = val_criterion(student_outputs, labels)
                student_top1, student_top5 = calculate_accuracy(student_outputs, labels, topk=(1, 5))
                classifier_pred = classifier(student_outputs)
                combined_labels = torch.zeros(batch_size, 1).to(device)
                binary_accuracy = calculate_binary_accuracy(classifier_pred, combined_labels)
                update_metrics(metrics, torch.tensor([-1]), torch.tensor([-1]), performance_loss, student_top1,
                               student_top5, binary_accuracy, mode='val')

        average_metrics(metrics, len(val_loader), mode='val')

        # Log training and validation metrics to TensorBoard
        log_metrics(writer, metrics, epoch, mode='train')
        log_metrics(writer, metrics, epoch, mode='val')

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Classifier Loss: {metrics["train_classifier_loss"]:.4f}, Train Top-1 Accuracy: {metrics["train_top1"]:.2f}%, Train Top-5 Accuracy: {metrics["train_top5"]:.2f}%,  Classifier Accuracy: {metrics["train_binary_accuracy"]:.2f}%')
        print(f'                  Val Loss: {metrics["val_student_loss"]:.4f}, Val Top-1 Accuracy: {metrics["val_top1"]:.2f}%, Val Top-5 Accuracy: {metrics["val_top5"]:.2f}%')
        if metrics["train_binary_accuracy"] < 0.5 and stoppage:
            print('Training is finished with accuracy {:.4f}'.format(binary_accuracy))
            print(
                f'Epoch {epoch + 1}/{num_epochs}, Train Classifier Loss: {metrics["train_classifier_loss"]:.4f}, Train Top-1 Accuracy: {metrics["train_top1"]:.2f}%, Train Top-5 Accuracy: {metrics["train_top5"]:.2f}%,  Classifier Accuracy: {metrics["train_binary_accuracy"]:.2f}%')
            print(
                f'                  Val Loss: {metrics["val_student_loss"]:.4f}, Val Top-1 Accuracy: {metrics["val_top1"]:.2f}%, Val Top-5 Accuracy: {metrics["val_top5"]:.2f}%')
            break

    # Save the models
    student_save_path = os.path.join(log_dir, f'{name_prefix}_student_model.pth')
    torch.save(student.state_dict(), student_save_path)
    print(f'Student model saved to {student_save_path}')
    classifier_save_path = os.path.join(log_dir, f'{name_prefix}_classifier_model.pth')
    torch.save(classifier.state_dict(), classifier_save_path)
    print(f'Classifier model saved to {classifier_save_path}')
    writer.close()

def train_classifier(X, y, **kwargs):
    u_classifier = LogisticRegression(input_dim=1, device='cpu')
    X = X.detach().cpu()
    y = y.detach().cpu()
    u_classifier.fit(X, y, device='cpu', **kwargs)
    predictions = u_classifier.predict(X)
    return predictions, u_classifier.linear

def untrain_models_contrastive(student, teacher, classifier, train_loader, val_loader, retain_loader, forget_loader, num_epochs=10, learning_rate=0.001,
                 log_dir='runs', device='cuda', name_prefix=get_current_datetime_string(),
                   freeze_classifier=False, freeze_student=False,
                   stoppage=False, reduction_size=16):
    # Define loss function and optimizers
    criterion = nn.BCELoss()
    negative_criterion = negative_loss(criterion)
    val_criterion = nn.CrossEntropyLoss()
    student_optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    #classifier_optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(os.path.join(log_dir, name_prefix))
    val_loader_ordered, forget_loader_ordered  = CorrespondingLoaders(val_loader, forget_loader, seed=42).get_loaders()
    val_loader_ordered, val_loader_ordered_2 = CorrespondingLoaders(val_loader_ordered, val_loader, seed=42).get_loaders()
    for epoch in range(num_epochs):
        student.train()
        classifier.train()
        metrics = initialize_metrics(mode='train')
        total_iterations = len(val_loader_ordered)
        for i, ((inputs_retain, labels_retain), (inputs_forget, labels_forget), (inputs_val, labels_val), (inputs_val_2, labels_val_2)) in enumerate(zip(retain_loader, forget_loader_ordered, val_loader_ordered, val_loader_ordered_2)):
            if i == total_iterations - 1:
                'Skipping last batch'
                print(inputs_forget.shape)
                i-=1
                break
            assert labels_val.eq(labels_forget).all()
            assert inputs_val.ne(inputs_forget).any()
            assert labels_val_2.eq(labels_val).all()
            assert inputs_val_2.ne(inputs_val).any()
            inputs_retain = inputs_retain.to(device)
            labels_retain = labels_retain.to(device)
            inputs_forget = inputs_forget.to(device)
            labels_forget = labels_forget.to(device)
            inputs_val = inputs_val.to(device)
            labels_val = labels_val.to(device)
            inputs_val_2 = inputs_val_2.to(device)
            labels_val_2 = labels_val_2.to(device)
            batch_size = inputs_forget.size(0)

            # Get outputs from student and teacher models for forget
            student_outputs_forget = student(inputs_forget)
            teacher_outputs_forget = teacher(inputs_forget)
            # Get outputs from student and teacher models for val
            student_outputs_val = student(inputs_val)
            teacher_outputs_val = teacher(inputs_val)
            # Outputs for different batch of val
            student_outputs_val_2 = student(inputs_val_2)
            teacher_outputs_val_2 = teacher(inputs_val_2)
            # Create labels for classifier
            # forget_labels = torch.ones(batch_size, 1).to(device)
            # val_labels = torch.zeros(batch_size, 1).to(device)

            # Shuffle inputs to create different pairs
            shuffled_indices = torch.randperm(2*batch_size)

            # Combine same and different pairs
            outputs_1 = classifier(student_outputs_forget, labels_forget)
            outputs_2 = classifier(teacher_outputs_val, labels_val)
            outputs_1 = outputs_1.view(batch_size // reduction_size, reduction_size, -1).mean(dim=1, keepdim=False)
            outputs_2 = outputs_2.view(batch_size // reduction_size, reduction_size, -1).mean(dim=1, keepdim=False)
            forget_labels = torch.ones(batch_size // reduction_size, 1).to(device)
            val_labels = torch.zeros(batch_size // reduction_size, 1).to(device)
            combined_outputs = torch.cat((outputs_1, outputs_2), dim=0)
            combined_labels = torch.cat((forget_labels, val_labels), dim=0)
            shuffled_indices = torch.randperm(2 * batch_size // reduction_size)
            # combined_outputs = combined_outputs[shuffled_indices]
            # combined_labels = combined_labels[shuffled_indices]
            # Train classifier
            if not freeze_classifier:
                classifier_optimizer.zero_grad()
                classifier_pred = classifier(combined_outputs_1.detach(), combined_outputs_2.detach())
                classifier_loss_value = criterion(classifier_pred, combined_labels)
                classifier_loss_value.backward()
                classifier_optimizer.step()

            # Train student to mimic teacher
            if not freeze_student:
                student_optimizer.zero_grad()
                predictions, linear_ = train_classifier(combined_outputs, combined_labels, epochs=1000, lr=0.01)
                u_classifier = LogisticRegression(input_dim=1, pretrained_model=linear_)
                predictions = u_classifier(combined_outputs)
                student_loss_value = negative_criterion(predictions, combined_labels)
                student_loss_value.backward()
                student_optimizer.step()

            with torch.no_grad():
                pred = student(inputs_retain)
                performance_loss = val_criterion(pred, labels_retain)
                student_top1, student_top5 = calculate_accuracy(pred, labels_retain, topk=(1, 5))
                acc_ = calculate_binary_accuracy(predictions, combined_labels, threshold=0.5)
                pred = student(inputs_forget)
                student_top1_forget, student_top5_forget = calculate_accuracy(pred, labels_forget, topk=(1, 5))
                update_metrics(metrics, -student_loss_value, torch.tensor([-1]),
                               performance_loss, student_top1,
                               student_top5, student_top1_forget,
                               student_top5_forget, acc_,
                               mode='train')

        average_metrics(metrics, i + 1, mode='train')
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
                performance_loss = val_criterion(student_outputs, labels)
                student_top1, student_top5 = calculate_accuracy(student_outputs, labels, topk=(1, 5))
                update_metrics(metrics, torch.tensor([-1]), torch.tensor([-1]), performance_loss, student_top1,
                               student_top5, torch.tensor([-1]), torch.tensor([-1]), torch.tensor([-1]), mode='val')

        average_metrics(metrics, len(val_loader), mode='val')

        # Log training and validation metrics to TensorBoard
        log_metrics(writer, metrics, epoch, mode='train')
        log_metrics(writer, metrics, epoch, mode='val')

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Classifier Loss: {metrics["train_student_loss"]:.4f}, Train Top-1 Accuracy: {metrics["train_top1"]:.2f}%, Train Top-5 Accuracy: {metrics["train_top5"]:.2f}%,  Train Top-1 Accuracy (Forget): {metrics["train_top1_forget"]:.2f}%, Train Top-5 Accuracy (Forget): {metrics["train_top5_forget"]:.2f}%,  Classifier Accuracy: {metrics["train_binary_accuracy"]:.2f}%')
        print(f'                  Val Loss: {metrics["val_student_loss"]:.4f}, Val Top-1 Accuracy: {metrics["val_top1"]:.2f}%, Val Top-5 Accuracy: {metrics["val_top5"]:.2f}%')
        if metrics["train_binary_accuracy"] < 0.50 and stoppage:
            print('Training is finished with accuracy {:.4f}'.format(binary_accuracy))
            print(
                f'Epoch {epoch + 1}/{num_epochs}, Train Classifier Loss: {metrics["train_classifier_loss"]:.4f}, Train Top-1 Accuracy: {metrics["train_top1"]:.2f}%, Train Top-5 Accuracy: {metrics["train_top5"]:.2f}%,  Classifier Accuracy: {metrics["train_binary_accuracy"]:.2f}%')
            print(
                f'                  Val Loss: {metrics["val_student_loss"]:.4f}, Val Top-1 Accuracy: {metrics["val_top1"]:.2f}%, Val Top-5 Accuracy: {metrics["val_top5"]:.2f}%')
            break

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
    if args.mode == 'train':
        classifier = ComparisonModel(input_dim=100, feature_size=64, temperature=0.1, use_linear=False, use_norm=False).to(device)
        # Train the models
        model = get_model(args.model, num_classes=100, pretrained_weights=None,
                          weight_path=args.weight_path)
        teacher = copy.deepcopy(model)
        for param in teacher.parameters():
            param.requires_grad = False
        student = get_model(args.model, num_classes=100, pretrained_weights=None, weight_path=args.weight_path_student)
        classifier.to(device)
        student.to(device)
        teacher.to(device)
        train_models(student, teacher, classifier, train_loader, val_loader, num_epochs=args.num_epochs,
                     learning_rate=args.learning_rate, log_dir=args.log_dir, device=device)
    elif args.mode == 'untrain':
        teacher = get_model(args.model, num_classes=100, pretrained_weights=None,
                          weight_path=args.weight_path)
        freeze_model(teacher)
        classifier = get_model('binary_classifier', num_classes=100, weight_path=args.weight_path_classifier)
        if args.freeze_classifier:
            freeze_model(classifier)
        student = get_model(args.model, num_classes=100, pretrained_weights=None,
                            weight_path=args.weight_path_student)
        if args.freeze_student:
            freeze_model(student)
        classifier.to(device)
        student.to(device)
        teacher.to(device)
        untrain_models_simple(student, teacher, classifier, train_loader, val_loader, retain_loader, forget_loader,
                              num_epochs=args.num_epochs, learning_rate=args.learning_rate,
                              log_dir=args.log_dir, device=device,
                              freeze_student=args.freeze_student, freeze_classifier=args.freeze_classifier,
                              name_prefix=args.name_prefix)
    elif args.mode == 'untrain_contrastive':
        teacher = get_model(args.model, num_classes=100, pretrained_weights=None,
                          weight_path=args.weight_path)
        freeze_model(teacher)
        classifier = nn.CrossEntropyLoss(reduction='none')
        # if args.freeze_classifier:
        #     freeze_model(classifier)
        student = get_model(args.model, num_classes=100, pretrained_weights=None,
                            weight_path=args.weight_path_student)
        if args.freeze_student:
            freeze_model(student)
        classifier.to(device)
        student.to(device)
        teacher.to(device)
        untrain_models_contrastive(student, teacher, classifier, train_loader, val_loader, retain_loader, forget_loader,
                              num_epochs=args.num_epochs, learning_rate=args.learning_rate,
                              log_dir=args.log_dir, device=device,
                              freeze_student=args.freeze_student, freeze_classifier=args.freeze_classifier,
                              name_prefix=args.name_prefix)

def build_name_prefix(args):
    prefix = ''
    if not args.untrain:
        prefix = f'model_{args.model}_sam_{args.use_sam}_rho_{args.rho}_lr_{args.learning_rate}_' + get_current_datetime_string()
    else:
        old_prefix = os.path.basename(args.weight_path)[:len('_model.pth')]
        prefix = 'untrained_' + old_prefix
    return prefix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a student model to mimic a teacher model with a classifier.")
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train for.')
    parser.add_argument('--data_dir', type=str, default='./data/cifar100', help='Directory to save dataset.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer.')
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory to save logs.')
    parser.add_argument('--config_dir', type=str, default='configs', help='Directory to save config.')
    parser.add_argument('--config_file', type=str, help='Path to configuration file to load.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training (e.g., "cpu", "cuda").')
    parser.add_argument('--name_prefix', type=str, default=None,
                        help='Define name prefix to store results (same prefix is used for logs, checkpoints, weights, etc).')
    parser.add_argument('--weight_path', type=str, help='Path to model weights file.')
    parser.add_argument('--weight_path_student', type=str, default=None, help='Path to model weights file to initialize student model.')
    parser.add_argument('--weight_path_classifier', type=str, default=None, help='Path to model weights file to initialize student model.')
    parser.add_argument('--model', type=str, default='resnet18', help='Model architecture to use.')
    parser.add_argument('--mode', type=str, default='train', help='Running mode: train/untrain/untrain_contrastive.')
    parser.add_argument('--freeze_classifier', action='store_true', help='Weather to freeze the classifier on the training.')
    parser.add_argument('--freeze_student', action='store_true', help='Weather to freeze the studenton the training.')

    args = parser.parse_args()
    if args.mode == 'untrain':
        assert args.weight_path_student is not None
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