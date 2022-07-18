import argparse
import pandas as pd


def read_csv(input_file):
    return pd.read_csv(input_file, index_col=False, encoding='utf-8')


def combine_data(df1, df2, labels):
    df_merged = pd.concat([df1, df2])
    df_merged.loc[~df_merged['dialect_norm'].isin(labels), 'dialect_norm'] = "OT"
    print(df_merged['dialect_norm'].value_counts())
    return df_merged


def undersample(df):
    classes = df.dialect_norm.value_counts().to_dict()
    least_class_amount = min(classes.values())
    classes_list = []
    for key in classes:
        classes_list.append(df[df['dialect_norm'] == key])
    classes_sample = []
    for i in range(0, len(classes_list)-1):
        classes_sample.append(classes_list[i].sample(least_class_amount))
    df_maybe = pd.concat(classes_sample)
    final_df = pd.concat([df_maybe, classes_list[-1]], axis=0)
    final_df = final_df.reset_index(drop=True)
    return final_df


def oversample(df, over_value=None):
    classes = df.dialect_norm.value_counts().to_dict()
    if over_value:
        most=over_value
    else:
        most = max(classes.values())
    classes_list = []
    for key in classes:
        classes_list.append(df[df['dialect_norm'] == key])
    classes_sample = []
    for i in range(1, len(classes_list)):
        classes_sample.append(classes_list[i].sample(most, replace=True))
    df_maybe = pd.concat(classes_sample)
    final_df = pd.concat([df_maybe, classes_list[0]], axis=0)
    final_df = final_df.reset_index(drop=True)
    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training data creator")
    parser.add_argument("--archimob", type=str, default="archimob_sentences_deduplicated.csv")
    parser.add_argument("--swissdial", type=str, default="swissdial_sentences.csv")
    parser.add_argument("--labels", type=list, default=["BE", "BS", "LU", "ZH", "DE", "OT"])
    parser.add_argument("--balance", type=str, choices=["undersample", "oversample"],default='undersample')
    # parser.add_argument("--balance", type=str, choices=["undersample", "oversample", "combined_resample"],default='oversample')

    args = parser.parse_args()

    archimob = read_csv(args.archimob)
    swissdial = read_csv(args.swissdial)

    df = combine_data(df1=archimob, df2=swissdial, labels=args.labels)

    if args.balance == 'undersample':
        training_data = undersample(df)
        print(training_data['dialect_norm'].value_counts())
    elif args.balance == 'oversample':
        training_data = oversample(df)
        print(training_data['dialect_norm'].value_counts())
    # elif args.balance == "combined_resample":
    #     training_data_temp = oversample(df, over_value=df[''])

    training_data.to_csv(f'training_data_archi_swiss_6_labels_{args.balance}.csv', index=None, encoding='utf-8')