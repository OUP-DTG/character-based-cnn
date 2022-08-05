"""
This process reads the files that have individual sentences from the commoncrawl corpus with dialect labels
assigned by the cnn model. It groups the individual sentences by document id (sha1) and disregards any labels that were
assigned with a confidence score smaller than 0.7. For the remaining sentence-label pairs, it assigns the majority label
to the whole amount of sentences in that document. If all the sentences in the grouping had a confidence score of < 0.7,
the full group of sentences is discarded from the dataset.
"""

import os

import pandas as pd


def read_files(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if not filename.endswith('.csv.gz'):
            continue
        print(filename)
        df = pd.read_csv(os.path.join(input_dir, filename))
        postprocessed_df = normalise_predictions(df)
        postprocessed_df.to_csv(os.path.join(output_dir, filename))


def normalise_predictions(df):
    processed_dfs_lst = []
    for idx, group in df.groupby('sha1'):
        # print(group)
        group_tmp = group[group['confidence'] > 0.7]
        if not group_tmp.empty:
            dom_label = group_tmp['pred_label'].value_counts().idxmax()
            group['majority_label'] = dom_label
            processed_dfs_lst.append(group)

    return pd.concat(processed_dfs_lst)


if __name__ == "__main__":
    INPUT_DIR = "different_sentence_split/dialect_classified_2017/"
    OUTPUT_DIR = "different_sentence_split/dialect_classified_2017/postprocessed"
    read_files(INPUT_DIR, OUTPUT_DIR)
