"""
This process is used to apply a different way of splitting our input sentences.
Instead of accepting the sentence splitting as provided by commoncrawl, we group the
sentences by document id (sha1), join the text, and perform a different split of text
chunks up to 150 characters long (requirement set by the cnn model), by also making sure we
don't split any words in the middle (textwrap module).
"""

import os
import textwrap

import pandas as pd


def read_files(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        print(filename)
        df = pd.read_csv(os.path.join(input_dir, filename))
        postprocessed_df = apply_chunk_splitting(df)
        print(postprocessed_df['sha1'].value_counts())
        postprocessed_df.to_csv(os.path.join(output_dir, filename))


def apply_chunk_splitting(df):
    print(df['sha1'].value_counts())
    processed_dfs_lst = []
    for idx, group in df.groupby('sha1'):
        tmp_text = ' '.join(group.text.values.tolist())
        chunks = textwrap.wrap(tmp_text, 150, break_long_words=False)
        tmp_df = pd.DataFrame(chunks, columns=['text'])
        tmp_df.insert(loc=0, column='sha1', value=group['sha1'].values.tolist()[0])
        tmp_df.insert(loc=0, column='snapshot', value='2017-05')
        processed_dfs_lst.append(tmp_df)
    return pd.concat(processed_dfs_lst, ignore_index=True)


if __name__ == "__main__":
    INPUT_DIR = "../data_filtered/2017"
    OUTPUT_DIR = "different_sentence_split/2017"
    read_files(INPUT_DIR, OUTPUT_DIR)
