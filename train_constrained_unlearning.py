import argparse
import yaml
import os
import copy
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from datetime import datetime
from models import get_model
import data
from tensorboard_settings import (
    get_current_datetime_string,
    initialize_metrics,
    update_metrics,
    update_additional_metrics,
    average_metrics,
    average_additional_metrics,
    log_metrics,
    log_metrics_unlearn,
    print_metrics,
    build_name_prefix,
    initialize_additional_metrics,
    log_additional_metrics,
    print_additional_metrics,
    log_metrics_unlearn_additional
)

from optimizer import (
    unlearning_stopping_criterion,
    refining_stopping_criterion,
    forget_forever_stopping_criterion,
    confusion_resolution_simple_stopping_criterion
    )

from metrics import calculate_accuracy, base_loss, compute_avg_loss

def _grad_norm(model):
    total_norm = 0.0

    for param in model.parameters():
        if param.grad is not None:
            param_norm = torch.norm(param.grad, 2.)
            total_norm += param_norm.item() ** 2.

    total_norm = total_norm ** .5
    return total_norm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def untrain_constrained(model,
                        model_teacher,
                        loss_fn_objective,
                        loader_objective,
                        loss_fn_condition,
                        loader_condition,
                        validloader,
                        internal_method,
                        num_epochs=10,
                        learning_rate=0.001,
                        log_dir='runs',
                        device='cuda',
                        name_prefix=get_current_datetime_string(),
                        stopping_criterion=None,
                        previous_val_loss=None,
                        epsilon_preset=False,
                        seed=42) -> None:
    """
    Unlearn a model based on the constrained problem formulation.

    Args:
        model: The neural network model that requires unlearning.
        model_teacher: The neural network model which might be 
            utilized for contrastive learning/unlearning purpose.
        loss_fn_objective: The loss function minimized in
            the constrained sense.
        loader_objective: The corresponding to `loss_objective` loader.
        loss_fn_condition: The loss function condition on which is kept in
            the constrained optimization problem.
        loader_condition: The corresponding to `loss_condition` loader.
        validloader: DataLoader for the validation data.
        internal_method: Method for solving the constrained optimization
            problem. Available options: `penalty`, `lagrange'.
        num_epochs: Number of epochs to untrain.
        log_dir: Directory to save TensorBoard logs.
        device: Device to run the model on (e.g., 'cpu' or 'cuda').
        stopping_criterion: If None, the algorithm runs for
            `num_epochs`. If 'unlearning', the algorithm stops when the forget loss
            gets larger than the validation loss; algorithm terminates not
            later than `num_epochs` epochs. If 'refining', the algorithm will stop
            if the validation loss has not decreased for the past 2 epochs.
    """

    set_seed(seed)

    reference_loss_value = None
    
    if name_prefix is None:
        name_prefix = get_current_datetime_string()

    name_prefix += f'_constrained_{internal_method}'

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

    epsilon_condition = torch.tensor(0.25).requires_grad_(False)
    alpha = 5.
    if stopping_criterion in ['unlearning', None, 'forget-forever']:
        dual_variable = torch.tensor(1.).requires_grad_(False).to(device)
    elif stopping_criterion in ['refining', 'confusion']:
        dual_variable = torch.tensor([1., 1.]).requires_grad_(False).to(device)
    else:
        raise ValueError('Unknown keyword')

    stop_training = False
    val_loss_decreased_at_the_previous_epoch = True
    
    if stopping_criterion == 'refining': # this might be a hack, too; fix it
        assert previous_val_loss is not None, \
            "Validation value must be pre-set"
        
    # Compute a randomly initialized model loss at request
    random_init_model_loss = None
    if stopping_criterion == 'confusion':
        print('Stopping criterion is confusion.')
        assert len(loader_condition) <= len(loader_objective)
        model_dumb = copy.deepcopy(model)
        for layer in model_dumb.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        model_dumb.to('cuda')
        # hack; nn.CrossEntropyLoss() and loss_fn_condition are related
        random_init_model_loss = compute_avg_loss(model_dumb, loader_condition, nn.CrossEntropyLoss()) 
        print(f'Randomly initialized model\'s loss on the forget loader is {random_init_model_loss}')
        assert type(random_init_model_loss) == float

    if stopping_criterion == 'confusion':
        reference_loss_value = random_init_model_loss
    if stopping_criterion == 'refining':
        reference_loss_value = previous_val_loss

    assert(type(reference_loss_value) == float)

    # Run unlearning epochs
    print('Make sure the forgetloader has smaller size the retainloader.')
    for epoch in range(num_epochs):
        if stop_training:
            break

        model.train()
        metrics_objective = initialize_additional_metrics(keys=['loss', 'top1', 'top5'])
        metrics_condition = initialize_additional_metrics(keys=['loss', 'top1', 'top5', 'loss_forget'])
        metrics_validation = initialize_additional_metrics(keys=['loss', 'top1', 'top5'])
        metrics_both = initialize_additional_metrics(keys=['grad'])
        n_batches = 0

        # Untraining loop
        assert len(loader_objective) > 0 and len(loader_condition) > 0

        if internal_method=='penalty':
            print(f'Current alpha = {alpha}')

        if internal_method=='lagrange':
            print(f'Current dual variable = {dual_variable.cpu()}')

        for batch_objective, batch_condition \
                in zip(loader_objective, loader_condition):
            n_batches += 1
            inputs_objective, labels_objective = batch_objective
            inputs_condition, labels_condition = batch_condition

            inputs_objective, labels_objective = inputs_objective.to(device), labels_objective.to(device)
            inputs_condition, labels_condition = inputs_condition.to(device), labels_condition.to(device)

            loss_objective = loss_fn_objective(model, model_teacher, inputs_objective, labels_objective)
            if stopping_criterion in ['unlearning', 'forget-forever', None]: # this is a hack, fix later
                loss_condition = loss_fn_condition(model, model_teacher, inputs_condition, labels_condition)
            elif stopping_criterion in ['refining', 'confusion']:
                loss_condition = loss_fn_condition(model, model_teacher, inputs_condition, labels_condition,
                                                   reference=reference_loss_value)
                loss_forget = base_loss(model, model_teacher, inputs_condition, labels_condition)
            else:
                raise ValueError('Unknown keyword')


            if epoch == 0 and not epsilon_preset:
                # epsilon_condition should be equal to an average over a batch
                epsilon_condition = (n_batches - 1) * epsilon_condition
                epsilon_condition += loss_condition.clone().detach().cpu()
                epsilon_condition /= n_batches
                print(f'Epsilon updated! {epsilon_condition.item()}')

            # penalty method
            if internal_method == 'penalty':
                # penalty = -1 * alpha * torch.log(epsilon - retain_loss) # logarithmic barrier; does not work with SGD
                penalty = torch.square(torch.max(torch.tensor(0), loss_condition - epsilon_condition)) # quadratic penalty
                # penalty = torch.pow(torch.max(torch.tensor(0), retain_loss - epsilon), 4) # quartic penalty
                # penalty = torch.exp(torch.max(torch.tensor(0), retain_loss - epsilon)) - 1.
                loss = loss_objective + alpha * penalty

                base_optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
                curr_grad_norm = _grad_norm(model)
                base_optimizer.step()

            # lagrange method
            if internal_method == 'lagrange':
                if stopping_criterion in ['unlearning', None, 'forget-forever']:
                    lagrange_objective = loss_objective + dual_variable * (loss_condition - epsilon_condition)
                elif stopping_criterion in ['refining', 'confusion']:
                    lagrange_objective = loss_objective + \
                        dual_variable[0] * (loss_condition - epsilon_condition) + \
                        dual_variable[1] * (- loss_condition - epsilon_condition)
                else:
                    raise ValueError('Unknown keyword')

                base_optimizer.zero_grad()
                lagrange_objective.backward()
                curr_grad_norm = _grad_norm(model)
                base_optimizer.step()

                if stopping_criterion in ['unlearning', None, 'forget-forever']:
                    post_update_loss_condition = loss_fn_condition(model, model_teacher, inputs_condition, labels_condition)
                elif stopping_criterion in ['refining', 'confusion']:
                    post_update_loss_condition = loss_fn_condition(model, model_teacher, inputs_condition, labels_condition,
                                                                   reference=reference_loss_value)
                else:
                    raise ValueError('Unknown keyword')

                with torch.no_grad():
                    if stopping_criterion in ['unlearning', None, 'forget-forever']:
                        dual_variable += learning_rate * (post_update_loss_condition - epsilon_condition)
                    elif stopping_criterion in ['refining', 'confusion']:
                        dual_variable[0] += learning_rate * (post_update_loss_condition - epsilon_condition) 
                        dual_variable[1] += learning_rate * (-post_update_loss_condition - epsilon_condition)
                        dual_variable[0] = torch.max(torch.tensor(0).requires_grad_(False), dual_variable[0])
                        dual_variable[1] = torch.max(torch.tensor(0).requires_grad_(False), dual_variable[1])
                    else:
                        raise ValueError('Unknown keyword')


            top1, top5 = calculate_accuracy(model(inputs_condition),
                                            labels_condition, topk=(1, 5))
            update_additional_metrics(metrics_condition,
                                      loss=loss_condition,
                                      top1=top1,
                                      top5=top5,
                                      )
            if stopping_criterion in ['refining', 'confusion']:
                update_additional_metrics(metrics_condition, loss_forget=loss_forget)
            top1, top5 = calculate_accuracy(model(inputs_objective),
                                            labels_objective, topk=(1, 5))
            update_additional_metrics(metrics_objective,
                                      loss=loss_objective,
                                      top1=top1,
                                      top5=top5
                                      )
            update_additional_metrics(metrics_both, grad=curr_grad_norm)
            

        average_additional_metrics(metrics_condition,
                                   n_batches,
                                   keys=['loss', 'top1', 'top5'])
        if stopping_criterion in ['refining', 'confusion']:
            average_additional_metrics(metrics_condition, n_batches, keys=['loss_forget'])

        average_additional_metrics(metrics_objective,
                                   n_batches,
                                   keys=['loss', 'top1', 'top5'])
        average_additional_metrics(metrics_both,
                                   n_batches,
                                   ['grad'])

        # penalty method updates
        if internal_method == 'penalty':
            if metrics_both['grad'] < 5. and epoch != 0:
                alpha *= 1.05

        # Validation loop
        model.eval()
        with torch.no_grad():
            for inputs, labels in validloader:
                # Move data to the appropriate device
                inputs, labels = inputs.to(device), labels.to(device)
                # Forward pass
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                top1, top5 = calculate_accuracy(outputs, labels, topk=(1, 5))
                update_additional_metrics(metrics_validation,
                                          loss=val_loss,
                                          top1=top1,
                                          top5=top5)

        average_additional_metrics(metrics_validation,
                                   len(validloader),
                                   keys=['loss', 'top1', 'top5'])
        
        print('Condition Set Performance')
        print_additional_metrics(metrics_condition, epoch, num_epochs, end=' ')
        print('Objective Set Performance')
        print_additional_metrics(metrics_objective, epoch, num_epochs, end=' ')
        print('Validation Set Performance')
        print_additional_metrics(metrics_validation, epoch, num_epochs)
        log_metrics_unlearn_additional(writer, metrics_objective, metrics_condition, metrics_validation, epoch, stopping_criterion)
        log_additional_metrics(writer, metrics_both, epoch, mode='', keys=['grad'])

        stop_training_1 = unlearning_stopping_criterion(stopping_criterion,
                                                      abs(metrics_objective['loss']),
                                                      abs(metrics_validation['loss'])
                                                      )
        
        (stop_training_2,
        val_loss_decreased_at_the_previous_epoch,
        previous_val_loss) = refining_stopping_criterion(
            stopping_criterion,
            val_loss_decreased_at_the_previous_epoch,
            metrics_validation['loss'],
            previous_val_loss,
            epoch,
            metrics_condition['loss'],
            epsilon_condition.item())
        
        stop_training_3 = forget_forever_stopping_criterion(stopping_criterion, metrics_condition['top5'])
        stop_training_4 = confusion_resolution_simple_stopping_criterion(stopping_criterion, loss_forget, random_init_model_loss)

        stop_training = stop_training_1 or stop_training_2 or stop_training_3 # or stop_training_4

    # Save the final model
    model_save_path = os.path.join(log_dir, f"{name_prefix}_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

    # Save model graph
    sample_inputs = next(iter(loader_condition))[0].to(device)
    writer.add_graph(model, sample_inputs)

    # Close the TensorBoard writer
    writer.close()
    if stopping_criterion in [None, 'unlearning']:
        print('Unlearning complete')
    elif stopping_criterion == 'refining':
        print('Refinement complete')
    elif stopping_criterion == 'forget-forever':
        print('Removal complete')        

    return previous_val_loss
        

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
                        args.constrained_internal_method,
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
    parser.add_argument('--constrained_internal_method', type=str, help='Internal constrained optimization problem.')


    args = parser.parse_args()

    if args.constrained_internal_method not in ['penalty', 'lagrange']:
        raise ValueError('Unknown value for `constrained_internal_method`.')

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