import pandas as pd

train_df = pd.read_csv("Dataset/train.tsv", sep="\t")["Headline"]
train_df.to_csv("Dataset/train.txt", index=False, header=False)

train_df = pd.read_csv("Dataset/test.tsv", sep="\t")["Headline"]
train_df.to_csv("Dataset/test.txt", index=False, header=False)
