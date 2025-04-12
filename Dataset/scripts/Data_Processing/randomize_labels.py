# read a tsv file, randomly shuffle the rows and overwrite the original file

import pandas as pd
import random
import os
from tqdm import tqdm

# Set the base directory where the TSV files are located
file = "Dataset/real/augmented/train.tsv"
base_dir = os.path.dirname(file)

# Read the TSV file into a DataFrame
df = pd.read_csv(file, sep="\t")
print(df.head())
print(df.columns)
print(df.shape)

# Shuffle the DataFrame rows randomly
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(df.head())
print(df.columns)
print(df.shape)
print("Randomized labels")

# Save the shuffled DataFrame back to the original TSV file
df.to_csv(file, sep="\t", index=False)
