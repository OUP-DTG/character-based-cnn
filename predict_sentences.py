import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data_loader import CLASS_TO_LABELS_MAP
from src.sentence_model import SentenceCNN, SWISS_GERMAN_ARCHIMOB_ALPHABET, SWISS_GERMAN_SWISSDIAL_ALPHABET, \
    COMBINED_ALPHABET
from src.utils import process_text, encode_string

use_cuda = torch.cuda.is_available()

def preprocess_input(args, input_file):

    number_of_characters = args.number_of_characters + len(args.extra_characters)
    identity_mat = np.identity(number_of_characters)
    vocabulary = list(args.alphabet) + list(args.extra_characters)
    max_length = args.max_length

    # chunk your dataframes in small portions
    chunks = pd.read_csv(
        input_file,
        chunksize=args.chunksize,
        encoding=args.encoding,
        # names=['text']
    )

    output_chunks = []
    for df_chunk in chunks:
        aux_df = df_chunk.copy()
        aux_df["processed_text"] = aux_df['text'].map(
            lambda text: process_text(args.steps, text)
        )

        aux_df["processed_text"] = aux_df['processed_text'].map(
            lambda text: encode_string(text, number_of_characters, identity_mat, vocabulary, max_length)
        )

        output_chunks.append(aux_df.copy())

    print(f"data loaded successfully with {len(output_chunks)} chunks")

    return output_chunks


def predict(args, input_file):

    output_columns = [CLASS_TO_LABELS_MAP[key] for key in sorted(CLASS_TO_LABELS_MAP.keys())]

    model = SentenceCNN(args, args.number_of_classes)
    state = torch.load(args.model)
    model.load_state_dict(state)
    model.eval()

    data_chunks = preprocess_input(args, input_file)

    for idx, chunk_df in enumerate(data_chunks):
        processed_input = torch.tensor(chunk_df["processed_text"].tolist())

        if use_cuda:
            processed_input = processed_input.to("cuda")
            model = model.to("cuda")

        prediction = model(processed_input)
        probabilities = F.softmax(prediction, dim=1)
        probabilities = probabilities.detach().cpu().numpy()
        pred_class = torch.max(prediction, 1)[1].cpu().numpy().tolist()
        pred_label = [CLASS_TO_LABELS_MAP[x] for x in pred_class]

        prob_df = pd.DataFrame(probabilities, columns=output_columns)
        chunk_df = pd.concat([chunk_df, prob_df.copy().reset_index(drop=True, inplace=True)], axis=1)
        chunk_df['pred_class'] = pred_class
        chunk_df['pred_label'] = pred_label
        chunk_df.drop(columns=['processed_text'], inplace=True)
        data_chunks[idx] = chunk_df

    return data_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Predict a pretrained Character Based CNN for text classification"
    )
    parser.add_argument("--model", type=str, help="path for pre-trained model")
    parser.add_argument("--data_path", type=str, help="directory to input data")
    parser.add_argument("--steps", nargs="+", default=["lower"])
    parser.add_argument("--chunksize", type=int, default=40000)
    parser.add_argument("--encoding", type=str, default="utf-8")
    parser.add_argument("--sep", type=str, default=",")

    # arguments needed for the prediction
    parser.add_argument("--input_alphabet", type=str, choices=['archimob', 'swissdial', 'both'])
    parser.add_argument("--extra_characters", type=str, default="")
    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--number_of_classes", type=int, default=4)
    parser.add_argument("--embeddings", action="store_true", help="flag to extract embeddings")
    # output
    # parser.add_argument("--output_csv", type=str, default="archimob_sentences.predict_test.random1K.predictions.csv")

    args = parser.parse_args()

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

    for filename in os.listdir(args.data_path):
        print(filename)
        filepath = os.path.join(args.data_path, filename)

        # is a list of dfs
        prediction_df = predict(args, filepath)

        # concat them all in a single df
        prediction_df = pd.concat(prediction_df)

        # save to file
        prediction_df.to_csv(f'dialect_classified_data/2017/dialect_class_{filename}', index=False, encoding=args.encoding)
