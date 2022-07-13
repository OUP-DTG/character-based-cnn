#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import shutil
import os

from types import SimpleNamespace
from datetime import datetime
from tensorboardX import SummaryWriter

import torch
from torch.utils.data import SubsetRandomSampler, DataLoader
from sklearn.model_selection import KFold

from src.data_loader import MyDataset, load_data
from src.sentence_model import SWISS_GERMAN_ARCHIMOB_ALPHABET, SWISS_GERMAN_SWISSDIAL_ALPHABET

from cli_arguments import read_cli_arguments
from train import run


def kfold_run(args, k_folds=5):

    texts, labels, number_of_classes, sample_weights = load_data(args)

    class_names = sorted(list(set(labels)))
    class_names = [str(class_name) for class_name in class_names]

    # For fold results
    results = {}

    # Set fixed random number seed
    torch.manual_seed(42)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Start print
    print("-" * 50)

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(texts, labels)):
        # Print
        print(f'FOLD {fold}')
        print("-" * 50)

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)

        full_dataset = MyDataset(texts, labels, args)

        # set up train/val parameters for DataLoader
        training_params = {
            "batch_size": args.batch_size,
            "shuffle": False,
            "num_workers": args.workers,
            "drop_last": True,
            "sampler": train_subsampler
        }

        validation_params = {
            "batch_size": args.batch_size,
            "shuffle": False,
            "num_workers": args.workers,
            "drop_last": True,
            "sampler": test_subsampler
        }

        training_generator = DataLoader(full_dataset, **training_params)
        validation_generator = DataLoader(full_dataset, **validation_params)

        kfold_args = SimpleNamespace(kfold=str(fold),
                                     number_of_classes=number_of_classes,
                                     class_names=class_names,
                                     training_generator=training_generator,
                                     validation_generator=validation_generator)

        k_best_f1, k_best_loss, k_best_acc, k_best_epoch = run(args, kfold_args, writer, log_file)
        results[fold] = dict(f1=k_best_f1,
                             loss=k_best_loss,
                             acc=k_best_acc,
                             epoch=k_best_epoch)
    return results


if __name__ == "__main__":

    args = read_cli_arguments()

    if args.input_alphabet == 'swissdial':
        setattr(args, 'alphabet', SWISS_GERMAN_SWISSDIAL_ALPHABET)
        setattr(args, 'number_of_characters', len(SWISS_GERMAN_SWISSDIAL_ALPHABET))
    elif args.input_alphabet == 'archimob':
        setattr(args, 'alphabet', SWISS_GERMAN_ARCHIMOB_ALPHABET)
        setattr(args, 'number_of_characters', len(SWISS_GERMAN_ARCHIMOB_ALPHABET))
    else:
        print("Wrong input alphabet value. Valid values are 'archimob' or 'swissdial'")
        sys.exit()

    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    # set log file
    if args.flush_history == 1:
        objects = os.listdir(args.log_path)
        for f in objects:
            if os.path.isdir(args.log_path + f):
                shutil.rmtree(args.log_path + f)

    now = datetime.now()
    logdir = args.log_path + now.strftime("%Y%m%d-%H%M%S") + "/"
    os.makedirs(logdir)
    log_file = logdir + "log.txt"
    writer = SummaryWriter(logdir)

    # prepare arguments common to all runs
    results = kfold_run(args, args.kfolds)

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {args.kfolds} FOLDS')
    print("-" * 50)
    acc_sum = 0.0
    loss_sum = 0.0
    f1_sum = 0.0
    for k_fold, k_result in results.items():
        print(f'Fold {k_fold}: Loss: {k_result["loss"]}, Acc. {k_result["acc"]}, F1: {k_result["f1"]} %')
        acc_sum += k_result["acc"]
        loss_sum += k_result["loss"]
        f1_sum += k_result["f1"]
    print(f'Average: Loss: {loss_sum/ len(results.items())}, Acc. {acc_sum/ len(results.items())}, F1: {f1_sum/ len(results.items())} %')

