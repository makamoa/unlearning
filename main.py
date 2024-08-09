import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
from models import get_model
import data
from metrics import (
    base_loss,
    negative_CE_loss,
    reference_loss
)

from tensorboard_settings import (
    build_name_prefix,
    load_config,
    save_config
)

from train import train_model
from untrain_baselines import (
    catastrophic_forgetting,
    exact_unlearning,
    finetuning,
    neggradplus,
    SCRUB
)

from train_constrained_unlearning import untrain_constrained

def main(args):
    # Define CIFAR100 dataset handler
    dataset_handler = data.CIFAR100Handler(batch_size=args.batch_size,
                                           validation_split=0.1,
                                           random_seed=42,
                                           data_dir=args.data_dir)
    # data_confuser = data.uniform_confuser(confuse_level=.0, random_seed=42)
    data_confuser = data.uniform_confuser(confuse_level=.1, random_seed=42)
    splitter = data.mix_both_sets(
        amend_split=1.,
        # retain_split=0.1,
        retain_split=0.,
        random_seed=42
        )
    confused_dataset_handler = data.AmendedDatasetHandler(
        dataset_handler,
        data_confuser,
        splitter
        )
    train_loader, val_loader, test_loader, forget_loader, \
    retain_loader, unseen_loader = confused_dataset_handler.get_dataloaders()

    print(f'Loader is loaded: {confused_dataset_handler.forget_retain_function}')

    # Initialize model
    model = get_model(args.model, num_classes=100, pretrained_weights=None,
                      weight_path=args.weight_path)
    device = torch.device(args.device if torch.cuda.is_available() or \
                                         'cpu' not in args.device else 'cpu')
    model.to(device)

    # Train the model
    if not args.untrain:
        current_train_loader = None
        if args.retain_set:
            print('The model is trained on the retain set.')
            current_train_loader = retain_loader
        else:
            print('The model is trained on the full train set.')
            current_train_loader = train_loader
        train_model(model,
                    current_train_loader,
                    val_loader,
                    num_epochs=args.num_epochs,
                    learning_rate=args.learning_rate,
                    log_dir=args.log_dir,
                    device=device,
                    use_sam=args.use_sam,
                    rho=args.rho,
                    name_prefix=args.name_prefix)
    else:
        if args.unlearning_algorithm == 'SCRUB':
            forget_optimizer = torch.optim.SGD(model.parameters(),
                                            lr=args.kl_forget_lr,
                                            momentum=0.9)
            forget_optimizer_stop = args.untrain_num_epochs
            print(f'SCRUB forget optimizer gets called off in {forget_optimizer_stop} epochs.')
            forget_scheduler = torch.optim.lr_scheduler.StepLR(
                forget_optimizer,
                step_size=forget_optimizer_stop,
                gamma=0.)
            retain_optimizer = torch.optim.SGD(model.parameters(),
                                            lr=args.kl_retain_lr,
                                            momentum=0.9)
            SCRUB(model=model,
                retainloader=retain_loader,
                forgetloader=forget_loader,
                validloader=val_loader,
                retain_optimizer=retain_optimizer,
                forget_optimizer=forget_optimizer,
                forget_scheduler=forget_scheduler,
                num_epochs=args.untrain_num_epochs,
                learning_rate=args.learning_rate,
                log_dir=args.log_dir,
                device=args.device,
                name_prefix=args.name_prefix,
                shot_epoch=None,
                stopping_criterion=args.stop_criterion)

        if args.unlearning_algorithm == 'neggradplus':
            forget_optimizer = torch.optim.SGD(model.parameters(),
                                            lr=args.kl_forget_lr,
                                            momentum=0.9)
            neggradplus(model=model,
                        retainloader=retain_loader,
                        forgetloader=forget_loader,
                        validloader=val_loader,
                        forget_optimizer=forget_optimizer,
                        num_epochs=args.untrain_num_epochs,
                        learning_rate=args.learning_rate,
                        log_dir=args.log_dir,
                        device=args.device,
                        name_prefix=args.name_prefix,
                        shot_epoch=None,
                        stopping_criterion=args.stop_criterion)
            
        if args.unlearning_algorithm == 'euk':
            exact_unlearning(model=model,
                            k=-1,
                            retainloader=retain_loader,
                            forgetloader=forget_loader,
                            validloader=val_loader,
                            num_epochs=args.finetuning_num_epochs,
                            learning_rate=args.learning_rate,
                            log_dir=args.log_dir,
                            device=args.device,
                            name_prefix=args.name_prefix,
                            shot_epoch=0,
                            stopping_criterion=args.stop_criterion)
            
        if args.unlearning_algorithm == 'cfk':
            catastrophic_forgetting(model=model,
                                    k=-1,
                                    retainloader=retain_loader,
                                    forgetloader=forget_loader,
                                    validloader=val_loader,
                                    num_epochs=args.finetuning_num_epochs,
                                    learning_rate=args.learning_rate,
                                    log_dir=args.log_dir,
                                    device=args.device,
                                    name_prefix=args.name_prefix,
                                    shot_epoch=0,
                                    stopping_criterion=args.stop_criterion)

        if args.unlearning_algorithm == 'constrained_unlearning':
            # unlearning in the constrained sense
            loss_fn_objective = negative_CE_loss
            loss_fn_condition = base_loss
            prev_val_loss = untrain_constrained(model=model,
                                                model_teacher=None,
                                                loss_fn_objective=loss_fn_objective,
                                                loader_objective=forget_loader,
                                                loss_fn_condition = loss_fn_condition,
                                                loader_condition=retain_loader,
                                                validloader=val_loader,
                                                internal_method=args.constrained_internal_method,
                                                num_epochs=args.untrain_num_epochs,
                                                learning_rate=args.learning_rate,
                                                log_dir=args.log_dir,
                                                device=args.device,
                                                name_prefix=args.name_prefix,
                                                stopping_criterion=args.stop_criterion)
            
        if args.unlearning_algorithm == 'constrained_learning':
            loss_fn_objective = base_loss
            loss_fn_condition = reference_loss
            full_name_prefix = args.name_prefix + \
                '_' + args.unlearning_algorithm + \
                '_' + args.constrained_internal_method + \
                '_' + args.stop_criterion
            print('Starting constrained learning.')
            if args.stop_criterion == 'confusion':
                prev_val_loss = None
                num_epochs = args.untrain_num_epochs
            elif args.stop_criterion == 'finetuning':
                num_epochs = args.finetuning_num_epochs
            untrain_constrained(model=model,
                                model_teacher=None,
                                loss_fn_objective=loss_fn_objective,
                                loader_objective=retain_loader,
                                loss_fn_condition=loss_fn_condition,
                                loader_condition=forget_loader,
                                validloader=val_loader,
                                internal_method='lagrange',
                                num_epochs=num_epochs,
                                learning_rate=args.learning_rate,
                                log_dir=args.log_dir,
                                device=args.device,
                                name_prefix=full_name_prefix,
                                stopping_criterion=args.stop_criterion,
                                previous_val_loss=prev_val_loss,
                                epsilon_preset=True)

        # if args.unlearning_algorithm in ['finetuning', 'SCRUB', 'neggradplus', 'constrained_unlearning']:
        #     # Once the unlearning is done, the model is finetuned
        #     print('Finetuning the model.')
        #     if args.unlearning_algorithm in ['SCRUB', 'neggradplus', 'constrained_unlearning']:
        #         print(f'After unlearning by {args.unlearning_algorithm}')
        #     full_name_prefix = args.name_prefix + '_' + args.unlearning_algorithm
        #     finetuning(model=model,
        #             retainloader=retain_loader,
        #             forgetloader=forget_loader,
        #             validloader=val_loader,
        #             num_epochs=args.finetuning_num_epochs,
        #             learning_rate=args.learning_rate,
        #             log_dir=args.log_dir,
        #             device=args.device,
        #             name_prefix=full_name_prefix,
        #             shot_epoch=0,
        #             stopping_criterion_enabled=args.stop_criterion)

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
    parser.add_argument('--use_sam', action='store_true', help='Whether to use SAM optimizer or not')
    parser.add_argument('--rho', type=float, default=None, help='SAM radius parameter')
    parser.add_argument('--name_prefix', type=str, default=None,
                        help='Define name prefix to store results (same prefix is used for logs, checkpoints, weights, etc).')
    parser.add_argument('--untrain', type=bool, default=False, help='Whether to untrain the model or perform the training')
    parser.add_argument('--retain_set', type=bool, default=False, help='Whether training must be done on the  scratch or not')
    parser.add_argument('--sam_lr', type=float, default=0.1, help='Learning rate for the SAM base optimizer')
    parser.add_argument('--kl_retain_lr', type=float, default=0.1,
                        help='Learning rate for the remaining part of the retain loss')
    parser.add_argument('--kl_forget_lr', type=float, default=0.1, help='Learning rate for the forget loss')
    parser.add_argument('--untrain_num_epochs', type=int, default=0, help='Number of epochs to untrain for.')
    parser.add_argument('--finetuning_num_epochs', type=int, default=0, help='Number of epochs for finetining the model.')
    parser.add_argument('--unlearning_algorithm', type=str, help='Sets the unlearning algorithm. Available options: \
                        [SCRUB, finetuning, euk, cfk, neggradplus, constrained_unlearning]')
    parser.add_argument('--stop_criterion', type=str, default=None, help='What stop criterion to apply. Options: `unlearning`, `refining`, `forget-forever`.')
    parser.add_argument('--constrained_internal_method', type=str, help='Internal constrained optimization problem.')



    args = parser.parse_args()
    if args.untrain:
        assert args.weight_path is not None

    if args.untrain and args.unlearning_algorithm not in \
        ['SCRUB', 'finetuning', 'euk', 'cfk', 'neggradplus', 'constrained_unlearning', 'constrained_learning']:
        raise ValueError('`unlearning_algorithm` value is not known')

    if args.unlearning_algorithm == 'finetuning' and args.finetuning_num_epochs == 0:
        raise UserWarning('`finetuning_num_epochs` is zero')

    if args.unlearning_algorithm in ['constrained_unlearning', 'constrained_learning'] and\
        args.constrained_internal_method not in ['penalty', 'lagrange']:
        raise ValueError('Unknown value for `constrained_internal_method`.')

    if args.retain_set:
        assert not args.untrain

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