import os

import pandas as pd


def read_files(input_dir, output_dir):
    for filename in os.listdir(input_dir):
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
    INPUT_DIR = "dialect_classified_data/2017"
    OUTPUT_DIR = "dialect_classified_data/postprocessed_2017"
    read_files(INPUT_DIR, OUTPUT_DIR)
