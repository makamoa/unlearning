import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
from models import get_model
import data
from optimizer import (
    SAM,
    base_loss,
    KL_retain_loss,
    KL_forget_loss,
    inverse_KL_forget_loss,
    NegativeCrossEntropyLoss,
    negative_CE_loss
)
import torch.optim.lr_scheduler as lr_scheduler

from train_model import train_model
from untrain_baselines import untrain_model
from helper_functions import (
    build_name_prefix,
    load_config,
    save_config
)

from train import train_model, untrain_model
from untrain_baselines import (
    catastrophic_forgetting,
    exact_unlearning,
    finetuning,
    neggradplus,
    SCRUB
)

def main(args):
    # Define CIFAR100 dataset handler
    dataset_handler = data.CIFAR100Handler(batch_size=args.batch_size,
                                           validation_split=0.1,
                                           random_seed=42,
                                           data_dir=args.data_dir)
    data_confuser = data.uniform_confuser(confuse_level=.0, random_seed=42)
    splitter = data.mix_both_sets(
        amend_split=1.,
        retain_split=0.1,
        random_seed=42
        )
    confused_dataset_handler = data.AmendedDatasetHandler(
        dataset_handler,
        data_confuser,
        splitter,
        class_wise_corr=True
        )
    train_loader, val_loader, test_loader, forget_loader, \
    retain_loader, unseen_loader = confused_dataset_handler.get_dataloaders()

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
        # forget_optimizer = torch.optim.SGD(model.parameters(),
        #                                    lr=args.kl_forget_lr,
        #                                    momentum=0.9)
        # forget_scheduler = torch.optim.lr_scheduler.StepLR(
        #     forget_optimizer,
        #     step_size=2,
        #     gamma=0.)
        # retain_optimizer = torch.optim.SGD(model.parameters(),
        #                                    lr=args.kl_retain_lr,
        #                                    momentum=0.9)
        # SCRUB(model=model,
        #       retainloader=retain_loader,
        #       forgetloader=forget_loader,
        #       validloader=val_loader,
        #       retain_optimizer=retain_optimizer,
        #       forget_optimizer=forget_optimizer,
        #       forget_scheduler=forget_scheduler,
        #       num_epochs=args.untrain_num_epochs,
        #       learning_rate=args.learning_rate,
        #       log_dir=args.log_dir,
        #       device=args.device,
        #       name_prefix=args.name_prefix)

        # forget_optimizer = torch.optim.SGD(model.parameters(),
        #                                    lr=args.kl_forget_lr,
        #                                    momentum=0.9)
        # forget_loss = KL_forget_loss
        # neggradplus(model=model,
        #             retainloader=retain_loader,
        #             forgetloader=forget_loader,
        #             validloader=val_loader,
        #             forget_loss=forget_loss,
        #             forget_optimizer=forget_optimizer,
        #             num_epochs=args.untrain_num_epochs,
        #             learning_rate=args.learning_rate,
        #             log_dir=args.log_dir,
        #             device=args.device,
        #             name_prefix=args.name_prefix)

        # finetuning(model=model,
        #            retainloader=retain_loader,
        #            forgetloader=forget_loader,
        #            validloader=val_loader,
        #            num_epochs=args.untrain_num_epochs,
        #            learning_rate=args.learning_rate,
        #            log_dir=args.log_dir,
        #            device=args.device,
        #            name_prefix=args.name_prefix)

        # exact_unlearning(model=model,
        #                  k=-5,
        #                  retainloader=retain_loader,
        #                  forgetloader=forget_loader,
        #                  validloader=val_loader,
        #                  num_epochs=args.untrain_num_epochs,
        #                  learning_rate=args.learning_rate,
        #                  log_dir=args.log_dir,
        #                  device=args.device,
        #                  name_prefix=args.name_prefix)

        catastrophic_forgetting(model=model,
                                k=-1,
                                retainloader=retain_loader,
                                forgetloader=forget_loader,
                                validloader=val_loader,
                                num_epochs=args.untrain_num_epochs,
                                learning_rate=args.learning_rate,
                                log_dir=args.log_dir,
                                device=args.device)

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