#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse


def read_cli_arguments():

    parser = argparse.ArgumentParser("Character Based CNN for text classification")
    # parser.add_argument("--data_path", type=str, default="archimob_sentences_deduplicated.csv")
    parser.add_argument("--data_path", type=str, default="training_data_archi_swiss_6_labels_undersample.csv")
    parser.add_argument("--validation_split", type=float, default=0.2)
    parser.add_argument("--label_column", type=str, default="dialect_norm")
    parser.add_argument("--text_column", type=str, default="sentence")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--encoding", type=str, default="utf-8")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--steps", nargs="+", default=["lower"])
    # parser.add_argument("--group_labels", type=int, default=0, choices=[0, 1])
    parser.add_argument("--group_labels", type=int, default=1, choices=[0, 1])
    parser.add_argument("--ignore_center", type=int, default=0, choices=[0, 1])
    # parser.add_argument("--label_ignored", type=list, default=['AG', 'GL', 'GR', 'NW', 'SG', 'SH', 'UR', 'VS', 'SZ', "DE"])
    parser.add_argument("--label_ignored", type=list, default=['AG', 'GL', 'GR', 'NW', 'SG', 'SH', 'UR', 'VS', 'SZ'])
    parser.add_argument("--ratio", type=float, default=1)
    parser.add_argument("--balance", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_sampler", type=int, default=0, choices=[0, 1])

    # parser.add_argument(
    #     "--alphabet",
    #     type=str,
    #     default="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\/\|_@#$%^&*~`+-=<>()[]{}",
    # )

    """
    parser.add_argument(
        "--alphabet",
        type=str,
        default=SWISS_GERMAN_ARCHIMOB_ALPHABET,
    )
    """

    parser.add_argument("--input_alphabet", type=str, choices=['archimob', 'swissdial', 'both'])
    # parser.add_argument("--number_of_characters", type=int, default=102)  # 95+7
    # parser.add_argument("--number_of_characters", type=int, default=NUM_SG_CHARS)
    # parser.add_argument("--extra_characters", type=str, default="äöüÄÖÜß")
    parser.add_argument("--extra_characters", type=str, default="")
    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--dropout_input", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)  # was 128
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default="sgd")
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--class_weights", type=int, default=0, choices=[0, 1])
    parser.add_argument("--focal_loss", type=int, default=0, choices=[0, 1])
    parser.add_argument("--gamma", type=float, default=2)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--drop_last", type=int, default=1, choices=[0, 1])

    parser.add_argument(
        "--scheduler", type=str, default="none", choices=["clr", "step", "none"]
    )
    # parser.add_argument("--min_lr", type=float, default=1.7e-3)
    parser.add_argument("--min_lr", type=float, default=2e-3)
    parser.add_argument("--max_lr", type=float, default=2e-2)
    # parser.add_argument("--max_lr", type=float, default=1e-2)
    parser.add_argument("--stepsize", type=float, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--early_stopping", type=int, default=0, choices=[0, 1])
    parser.add_argument("--checkpoint", type=int, choices=[0, 1], default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--log_path", type=str, default="./logs/")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--log_f1", type=int, default=1, choices=[0, 1])
    parser.add_argument("--flush_history", type=int, default=1, choices=[0, 1])
    parser.add_argument("--output", type=str, default="./models/")
    parser.add_argument("--model_name", type=str, default="test_model")
    parser.add_argument("--embeddings", action="store_true", help="flag to extract embeddings")
    parser.add_argument("--kfolds", type=int, default=5)

    args = parser.parse_args()

    return args
