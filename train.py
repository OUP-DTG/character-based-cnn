#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import shutil
import os

from types import SimpleNamespace
from datetime import datetime
from collections import Counter

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tensorboardX import SummaryWriter

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from src.data_loader import MyDataset, load_data
from src import utils
from src.sentence_model import SentenceCNN, SWISS_GERMAN_ARCHIMOB_ALPHABET, SWISS_GERMAN_SWISSDIAL_ALPHABET, COMBINED_ALPHABET
from src.focal_loss import FocalLoss

from cli_arguments import read_cli_arguments

def train(
        model,
        training_generator,
        optimizer,
        criterion,
        epoch,
        writer,
        log_file,
        scheduler,
        class_names,
        args,
        print_every=25,
):
    model.train()
    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()
    num_iter_per_epoch = len(training_generator)

    progress_bar = tqdm(enumerate(training_generator), total=num_iter_per_epoch)

    y_true = []
    y_pred = []

    for iter, batch in progress_bar:
        features, labels = batch
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        predictions = model(features)

        y_true += labels.cpu().numpy().tolist()
        y_pred += torch.max(predictions, 1)[1].cpu().numpy().tolist()

        loss = criterion(predictions, labels)

        loss.backward()
        if args.scheduler == "clr":
            scheduler.step()

        optimizer.step()
        training_metrics = utils.get_evaluation(
            labels.cpu().numpy(),
            predictions.cpu().detach().numpy(),
            list_metrics=["accuracy", "f1"],
        )

        losses.update(loss.data, features.size(0))
        accuracies.update(training_metrics["accuracy"], features.size(0))

        f1 = training_metrics["f1"]

        writer.add_scalar("Train/Loss", loss.item(), epoch * num_iter_per_epoch + iter)

        writer.add_scalar(
            "Train/Accuracy",
            training_metrics["accuracy"],
            epoch * num_iter_per_epoch + iter,
        )

        writer.add_scalar("Train/f1", f1, epoch * num_iter_per_epoch + iter)

        lr = optimizer.state_dict()["param_groups"][0]["lr"]

        if (iter % print_every == 0) and (iter > 0):
            print(
                "[Training - Epoch: {}], LR: {} , Iteration: {}/{} , Loss: {}, Accuracy: {}".format(
                    epoch + 1, lr, iter, num_iter_per_epoch, losses.avg, accuracies.avg
                )
            )

            if bool(args.log_f1):
                intermediate_report = classification_report(
                    y_true, y_pred, output_dict=True
                )

                f1_by_class = "F1 Scores by class: "
                for class_name in class_names:
                    f1_by_class += f"{class_name} : {np.round(intermediate_report[class_name]['f1-score'], 4)} |"

                print(f1_by_class)

    f1_train = f1_score(y_true, y_pred, average="weighted")

    writer.add_scalar("Train/loss/epoch", losses.avg, epoch + iter)
    writer.add_scalar("Train/acc/epoch", accuracies.avg, epoch + iter)
    writer.add_scalar("Train/f1/epoch", f1_train, epoch + iter)

    report = classification_report(y_true, y_pred)
    print(report)

    with open(log_file, "a") as f:
        f.write(f"Training on Epoch {epoch} \n")
        f.write(f"Average loss: {losses.avg.item()} \n")
        f.write(f"Average accuracy: {accuracies.avg.item()} \n")
        f.write(f"F1 score: {f1_train} \n\n")
        f.write(report)
        f.write("*" * 25)
        f.write("\n")

    return losses.avg.item(), accuracies.avg.item(), f1_train


def evaluate(
        model, validation_generator, criterion, epoch, writer, log_file, print_every=25
):
    model.eval()
    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()
    num_iter_per_epoch = len(validation_generator)

    y_true = []
    y_pred = []

    for iter, batch in tqdm(enumerate(validation_generator), total=num_iter_per_epoch):
        features, labels = batch
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
        with torch.no_grad():
            predictions = model(features)
        loss = criterion(predictions, labels)

        y_true += labels.cpu().numpy().tolist()
        y_pred += torch.max(predictions, 1)[1].cpu().numpy().tolist()

        validation_metrics = utils.get_evaluation(
            labels.cpu().numpy(),
            predictions.cpu().detach().numpy(),
            list_metrics=["accuracy", "f1"],
        )
        accuracy = validation_metrics["accuracy"]
        f1 = validation_metrics["f1"]

        losses.update(loss.data, features.size(0))
        accuracies.update(validation_metrics["accuracy"], features.size(0))

        writer.add_scalar("Test/Loss", loss.item(), epoch * num_iter_per_epoch + iter)

        writer.add_scalar("Test/Accuracy", accuracy, epoch * num_iter_per_epoch + iter)

        writer.add_scalar("Test/f1", f1, epoch * num_iter_per_epoch + iter)

        if (iter % print_every == 0) and (iter > 0):
            print(
                "[Validation - Epoch: {}] , Iteration: {}/{} , Loss: {}, Accuracy: {}".format(
                    epoch + 1, iter, num_iter_per_epoch, losses.avg, accuracies.avg
                )
            )

    f1_test = f1_score(y_true, y_pred, average="weighted")

    writer.add_scalar("Test/loss/epoch", losses.avg, epoch + iter)
    writer.add_scalar("Test/acc/epoch", accuracies.avg, epoch + iter)
    writer.add_scalar("Test/f1/epoch", f1_test, epoch + iter)

    report = classification_report(y_true, y_pred)
    print(report)

    with open(log_file, "a") as f:
        f.write(f"Validation on Epoch {epoch} \n")
        f.write(f"Average loss: {losses.avg.item()} \n")
        f.write(f"Average accuracy: {accuracies.avg.item()} \n")
        f.write(f"F1 score {f1_test} \n\n")
        f.write(report)
        f.write("=" * 50)
        f.write("\n")

    return losses.avg.item(), accuracies.avg.item(), f1_test


def run(args, dataloader_args, writer, log_file):

    model = SentenceCNN(args, dataloader_args.number_of_classes)
    if torch.cuda.is_available():
        model.cuda()

    if not bool(args.focal_loss):
        if bool(args.class_weights):
            class_counts = dict(Counter(dataloader_args.training_generator.dataset.labels))
            m = max(class_counts.values())
            for c in class_counts:
                class_counts[c] = m / class_counts[c]
            weights = []
            for k in sorted(class_counts.keys()):
                weights.append(class_counts[k])

            weights = torch.Tensor(weights)
            if torch.cuda.is_available():
                weights = weights.cuda()
                print(f"passing weights to CrossEntropyLoss : {weights}")
                criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()

    else:
        if args.alpha is None:
            criterion = FocalLoss(gamma=args.gamma, alpha=None)
        else:
            criterion = FocalLoss(
                gamma=args.gamma, alpha=[args.alpha] * dataloader_args.number_of_classes
            )

    if args.optimizer == "sgd":
        if args.scheduler == "clr":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=1, momentum=0.9, weight_decay=0.00001
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=args.learning_rate, momentum=0.9
            )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        print("f'unrecognized optimizer {}".format(args.optimizer))
        sys.exit()

    if args.scheduler == "clr":
        stepsize = int(args.stepsize * len(dataloader_args.training_generator))
        clr = utils.cyclical_lr(stepsize, args.min_lr, args.max_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
    else:
        scheduler = None

    best_f1 = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        training_loss, training_accuracy, train_f1 = train(
            model,
            dataloader_args.training_generator,
            optimizer,
            criterion,
            epoch,
            writer,
            log_file,
            scheduler,
            dataloader_args.class_names,
            args,
            args.log_every,
        )

        validation_loss, validation_accuracy, validation_f1 = evaluate(
            model,
            dataloader_args.validation_generator,
            criterion,
            epoch,
            writer,
            log_file,
            args.log_every,
        )

        print(
            "[Epoch: {} / {}]\ttrain_loss: {:.4f} \ttrain_acc: {:.4f} \tval_loss: {:.4f} \tval_acc: {:.4f}".format(
                epoch + 1,
                args.epochs,
                training_loss,
                training_accuracy,
                validation_loss,
                validation_accuracy,
            )
        )
        print("=" * 50)

        # learning rate scheduling

        # leave learning rate constant
        if args.scheduler == "step":
            if args.optimizer == "sgd" and ((epoch + 1) % 3 == 0) and epoch > 0:
                current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
                current_lr /= 2
                print("Decreasing learning rate to {0}".format(current_lr))
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

        # model checkpoint
        if validation_f1 > best_f1:
            best_f1 = validation_f1
            best_loss = validation_loss
            best_acc = validation_accuracy
            best_epoch = epoch
            if args.checkpoint == 1:
                checkpoint_filename = "model_{}_kfold{}_epoch_{}_maxlen_{}_lr_{}_loss_{}_acc_{}_f1_{}.pth".format(
                                        args.model_name,
                                        dataloader_args.kfold,
                                        epoch,
                                        args.max_length,
                                        optimizer.state_dict()["param_groups"][0]["lr"],
                                        round(validation_loss, 4),
                                        round(validation_accuracy, 4),
                                        round(validation_f1, 4))
                torch.save(model.state_dict(), os.path.join(args.output, checkpoint_filename))

        if bool(args.early_stopping):
            if epoch - best_epoch > args.patience > 0:
                print(
                    "Stop training at epoch {}. The lowest loss achieved is {} at epoch {}".format(
                        epoch, validation_loss, best_epoch
                    )
                )
                break

    return best_f1, best_loss, best_acc, best_epoch


def load_data_setup(args):

    # load data
    texts, labels, number_of_classes, sample_weights = load_data(args)

    class_names = sorted(list(set(labels)))
    class_names = [str(class_name) for class_name in class_names]

    # split data
    (
        train_texts,
        val_texts,
        train_labels,
        val_labels,
        train_sample_weights,
        _,
    ) = train_test_split(
        texts,
        labels,
        sample_weights,
        test_size=args.validation_split,
        random_state=42,
        stratify=labels,
    )

    training_set = MyDataset(train_texts, train_labels, args)
    validation_set = MyDataset(val_texts, val_labels, args)

    # set up train/val parameters for DataLoader
    training_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.workers,
        "drop_last": bool(args.drop_last),
    }

    validation_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "drop_last": bool(args.drop_last),
    }

    if bool(args.use_sampler):
        train_sample_weights = torch.from_numpy(train_sample_weights)
        sampler = WeightedRandomSampler(
            train_sample_weights.type("torch.DoubleTensor"), len(train_sample_weights)
        )
        training_params["sampler"] = sampler
        training_params["shuffle"] = False

    training_generator = DataLoader(training_set, **training_params)
    validation_generator = DataLoader(validation_set, **validation_params)

    setup_args = SimpleNamespace(kfold='',
                                 number_of_classes=number_of_classes,
                                 class_names=class_names,
                                 training_generator=training_generator,
                                 validation_generator=validation_generator)
    return setup_args


if __name__ == "__main__":

    args = read_cli_arguments()

    if args.input_alphabet == 'swissdial':
        setattr(args, 'alphabet', SWISS_GERMAN_SWISSDIAL_ALPHABET)
        setattr(args, 'number_of_characters', len(SWISS_GERMAN_SWISSDIAL_ALPHABET))
    elif args.input_alphabet == 'archimob':
        setattr(args, 'alphabet', SWISS_GERMAN_ARCHIMOB_ALPHABET)
        setattr(args, 'number_of_characters', len(SWISS_GERMAN_ARCHIMOB_ALPHABET))
    elif args.input_alphabet == 'both':
        setattr(args, 'alphabet', COMBINED_ALPHABET)
        setattr(args, 'number_of_characters', len(COMBINED_ALPHABET))
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

    # prepare arguments common to all runs and load data
    loaddata_args = load_data_setup(args)

    # run one split
    run(args, loaddata_args, writer, log_file)
