import sys
import argparse

import torch
import torch.nn.functional as F
import numpy as np

from src.sentence_model import SentenceCNN, SWISS_GERMAN_ARCHIMOB_ALPHABET, SWISS_GERMAN_SWISSDIAL_ALPHABET
from src.utils import preprocessing_steps

use_cuda = torch.cuda.is_available()


def preprocess_input(args):
    raw_text = args.text
    steps = args.steps
    for step in steps:
        raw_text = preprocessing_steps[step](raw_text)

    number_of_characters = args.number_of_characters + len(args.extra_characters)
    identity_mat = np.identity(number_of_characters)
    vocabulary = list(args.alphabet) + list(args.extra_characters)
    max_length = args.max_length

    processed_output = np.array(
        [
            identity_mat[vocabulary.index(i)]
            for i in list(raw_text[::-1])
            if i in vocabulary
        ],
        dtype=np.float32,
    )
    if len(processed_output) > max_length:
        processed_output = processed_output[:max_length]
    elif 0 < len(processed_output) < max_length:
        processed_output = np.concatenate(
            (
                processed_output,
                np.zeros(
                    (max_length - len(processed_output), number_of_characters),
                    dtype=np.float32,
                ),
            )
        )
    elif len(processed_output) == 0:
        processed_output = np.zeros(
            (max_length, number_of_characters), dtype=np.float32
        )
    return processed_output


def predict(args):

    model = SentenceCNN(args, args.number_of_classes)
    state = torch.load(args.model)
    model.load_state_dict(state)
    model.eval()

    processed_input = preprocess_input(args)

    processed_input = torch.tensor(processed_input)
    processed_input = processed_input.unsqueeze(0)
    if use_cuda:
        processed_input = processed_input.to("cuda")
        model = model.to("cuda")

    embeddings, prediction = model(processed_input)
    probabilities = F.softmax(prediction, dim=1)
    probabilities = probabilities.detach().cpu().numpy()
    return probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Predict a pretrained Character Based CNN for text classification"
    )
    parser.add_argument("--model", type=str, help="path for pre-trained model")
    parser.add_argument("--data_path", type=str, default="archimob_sentences_deduplicated.csv")
    parser.add_argument("--steps", nargs="+", default=["lower"])

    # arguments needed for the predicition
    parser.add_argument("--input_alphabet", type=str, choices=['archimob', 'swissdial'])
    parser.add_argument("--extra_characters", type=str, default="")
    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--number_of_classes", type=int, default=4)
    parser.add_argument("--embeddings", action="store_true", help="flag to extract embeddings")

    args = parser.parse_args()

    if args.input_alphabet == 'swissdial':
        setattr(args, 'alphabet', SWISS_GERMAN_SWISSDIAL_ALPHABET)
        setattr(args, 'number_of_characters', len(SWISS_GERMAN_SWISSDIAL_ALPHABET))
    elif args.input_alphabet == 'archimob':
        setattr(args, 'alphabet', SWISS_GERMAN_ARCHIMOB_ALPHABET)
        setattr(args, 'number_of_characters', len(SWISS_GERMAN_ARCHIMOB_ALPHABET))
    else:
        print("Wrong input alphabet value. Valid values are 'archimob' or 'swissdial'")
        sys.exit()

    prediction = predict(args)

    print("input : {}".format(args.text))
    print("prediction : {}".format(prediction))
