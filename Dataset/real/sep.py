import os
import shutil
import pandas as pd

tsv_file = "C:\pandith\output\synthetic-handwritten\\test.tsv"
source_folder = "output\synthetic-handwritten"
train_folder = "train"
test_folder = "test"

df = pd.read_csv(tsv_file, sep="\t")

df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

split_index = int(0.8 * len(df_shuffled))

train_df = df_shuffled.iloc[:split_index]
test_df = df_shuffled.iloc[split_index:]

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)


def copy_images(df_subset, dest_folder):
    for _, row in df_subset.iterrows():
        image_filename = row["file"]
        source_image_path = os.path.join(source_folder, image_filename)
        dest_image_path = os.path.join(dest_folder, image_filename)
        if os.path.exists(source_image_path):
            shutil.copy(source_image_path, dest_image_path)
        else:
            print(f"Warning: {source_image_path} does not exist.")


copy_images(train_df, train_folder)
copy_images(test_df, test_folder)

train_df.to_csv("train_labels.tsv", sep="\t", index=False)
test_df.to_csv("test_labels.tsv", sep="\t", index=False)

print("Images have been copied and TSV files created successfully.")
