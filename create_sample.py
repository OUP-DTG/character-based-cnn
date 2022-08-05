import pandas as pd
import os


def read_files(input_dir):
    list_of_dfs = []
    for filename in os.listdir(input_dir):
        # print(filename)
        list_of_dfs.append(pd.read_csv(os.path.join(input_dir, filename)))
    return pd.concat(list_of_dfs, ignore_index=True)


def sample_with_equal_distribution(df, method: str):
    dfs_list = []
    for idx, group in df.groupby('majority_label'):
        # ignore rows with digits for this sample
        group = group[~group['text'].str.contains(r'\d')]
        dfs_list.append(group.sample(n=250))
    merged_df = pd.concat(dfs_list, ignore_index=True)
    merged_df['method'] = method
    return merged_df


if __name__ == "__main__":
    INPUT_DIR_METHOD1 = "dialect_classified_data/postprocessed_2017"
    INPUT_DIR_METHOD2 = "different_sentence_split/dialect_classified_2017/postprocessed"
    sample1 = sample_with_equal_distribution(read_files(INPUT_DIR_METHOD1), 'method1')
    # print(sample1['majority_label'].value_counts())
    sample2 = sample_with_equal_distribution(read_files(INPUT_DIR_METHOD2), 'method2')
    # print(sample2['majority_label'].value_counts())

    final_df = pd.concat((sample1, sample2)).sample(frac=1)
    final_df.to_excel('dialect_classifier_sample_3000.xlsx')
