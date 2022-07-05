import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

from src.data_loader import MyDataset, load_data
from src.sentence_model import SentenceCNN


def extract_embeddings_from_trained_model(args):
    def _up_n(path, n):
        components = os.path.normpath(path).split(os.sep)
        return os.sep.join(components[:-n])

    model = SentenceCNN(args, args.number_of_classes)
    state = torch.load(os.path.join(_up_n(os.path.dirname(__file__), 1), 'models', args.model))
    model.load_state_dict(state)

    embeddings = []

    batch_size = 128
    workers = 1

    data_params = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": workers,
    }

    texts, labels, number_of_classes, sample_weights = load_data(args)
    sentences_set = MyDataset(texts, labels, args)

    data_generator = DataLoader(sentences_set, **data_params)

    model.eval()
    model.share_memory()  # NOTE: this is required for the ``fork`` method to work
    with torch.no_grad():
        for idx, batch in enumerate(data_generator):
            features, labels = batch
            temp_outputs = model(features)
            # embeddings.append(temp_outputs)
            embeddings.append([temp_outputs, labels])

    print(embeddings)

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    output_filename = f"{args.output_folder}/tensors.full.pt"
    save_tensors(embeddings, output_filename)
    print(len(embeddings))
    # the above returns 285, which is the number of batches consisting of 128 data points each (last one is smaller),
    # creating a total of 36,391 data points/vectors
    # each row/data point is a vector of 4 dimensions e.g. [0.8045, -1.5245, 0.6591, -0.0142]
    sys.exit()
    return embeddings


def save_tensors(tensors, output_filename):
    torch.save(tensors, output_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Testing a pretrained Character Based CNN for text classification"
    )
    parser.add_argument("--model", type=str, help="path for pre-trained model",
                        default="model_test_model_epoch_9_maxlen_150_lr_0.01_loss_0.7056_acc_0.7302_f1_0.7184.pth")

    # arguments needed for the model
    parser.add_argument(
        "--alphabet",
        type=str,
        default="() *,-./0123456789?ABCDEFGHIJKLMNOPRSTUVWZ_abcdefghijklmnoprstuvwxyzàáãäèéìíòóõöùúüĩǜ̀́ẽ",
    )
    parser.add_argument("--number_of_characters", type=int, default=88)
    parser.add_argument("--extra_characters", type=str, default="")
    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--number_of_classes", type=int, default=4)
    parser.add_argument("--data_path", type=str, default="../archimob_sentences_deduplicated.csv")
    parser.add_argument("--label_column", type=str, default="dialect_norm")
    parser.add_argument("--text_column", type=str, default="sentence")
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--encoding", type=str, default="utf-8")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--steps", nargs="+", default=["lower"])
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--group_labels", type=int, default=1, choices=[0, 1])
    parser.add_argument("--ignore_center", type=int, default=0, choices=[0, 1])
    parser.add_argument("--label_ignored", type=list, default=['AG', 'GL', 'GR', 'NW', 'SG', 'SH', 'UR', 'VS', 'SZ'])
    parser.add_argument("--ratio", type=float, default=1)
    parser.add_argument("--balance", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_sampler", type=int, default=0, choices=[0, 1])
    parser.add_argument("--output_folder", type=str, help="path for pre-trained embeddings", default="embeddings")

    args = parser.parse_args()
    embeddings = extract_embeddings_from_trained_model(args)
    print(embeddings)
