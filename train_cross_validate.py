#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import shutil
import os

from types import SimpleNamespace
from datetime import datetime
from tensorboardX import SummaryWriter

from torch.utils.data import SubsetRandomSampler, DataLoader, WeightedRandomSampler
from sklearn.model_selection import KFold

from src.data_loader import load_data
from src.sentence_model import SWISS_GERMAN_ARCHIMOB_ALPHABET, SWISS_GERMAN_SWISSDIAL_ALPHABET

from cli_arguments import read_cli_arguments
from train import run


def run_setup(k_folds=5):

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

    batch_size = args.batch_size

    training_params = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": args.workers,
        "drop_last": True,
    }

    validation_params = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "drop_last": True,
    }

    texts, labels, number_of_classes, sample_weights = load_data(args)

    class_names = sorted(list(set(labels)))
    class_names = [str(class_name) for class_name in class_names]

    # For fold results
    results = {}

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(texts, labels)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = DataLoader(dataset, batch_size=10, sampler=train_subsampler)
        testloader = DataLoader(dataset, batch_size=10, sampler=test_subsampler)

        setup_args = SimpleNamespace(writer=writer,
                                     log_file=log_file,
                                     number_of_classes=number_of_classes,
                                     class_names=class_names,
                                     sample_weights=sample_weights,
                                     texts=texts,
                                     labels=labels,
                                     training_params=training_params,
                                     validation_params=validation_params)
        yield setup_args


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

    # prepare arguments common to all runs
    setup_args = run_setup()

    # run one split
    run(args, setup_args)
