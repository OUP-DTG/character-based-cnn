import argparse
import os

import nltk
import torch
from nltk.cluster import KMeansClusterer
from sklearn import cluster, metrics

from src.sentence_model import SentenceCNN


def run(args):
    model = SentenceCNN(args, args.number_of_classes)

    state = torch.load(os.path.join('models', args.model))
    model.load_state_dict(state)

    x = model[model.vocab]

    print(x)

    num_clusters = 4

    kclusterer = KMeansClusterer(num_clusters, distance=nltk.cluster.util.cosine_distance, repeats=25)
    assigned_clusters = kclusterer.cluster(x, assign_clusters=True)
    print(assigned_clusters)

    words = list(model.vocab)
    for i, word in enumerate(words):
        print(f"{word}: {str(assigned_clusters[i])}")

    kmeans = cluster.KMeans(n_clusters=num_clusters)
    kmeans.fit(x)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    print("Cluster id for input data")
    print(labels)
    print("Centroids data")
    print(centroids)
    print("Score (opposite of the value of X on the K-means objective which is sum of distances of samples to their closest cluster centre):")
    print(kmeans.score(x))

    silhouette_score = metrics.silhouette_score(x, labels, metric="euclidean")
    print(f"Silhouete_score: {silhouette_score}")


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
    # parser.add_argument("--data_path", type=str, default="../archimob_sentences_deduplicated.csv")
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

    args = parser.parse_args()
    run(args)