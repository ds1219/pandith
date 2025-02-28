import os
import shutil
import pandas as pd

# Define file and folder names
tsv_file = 'Dataset/real/labels.tsv'       # Your original TSV file (with header "ile\tlabels")
source_folder = 'Dataset/real/preprocessed'   # Folder containing the original images
train_folder = 'train'
test_folder = 'test'

# Read the TSV file into a pandas DataFrame
df = pd.read_csv(tsv_file, sep='\t')

# Shuffle the DataFrame; random_state ensures reproducibility
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Calculate the split index for 80% training and 20% testing
split_index = int(0.8 * len(df_shuffled))

# Split the DataFrame into training and testing subsets
train_df = df_shuffled.iloc[:split_index]
test_df = df_shuffled.iloc[split_index:]

# Create destination folders for images if they do not exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Function to copy images from the source folder to the destination folder
def copy_images(df_subset, dest_folder):
    for _, row in df_subset.iterrows():
        image_filename = row['file']
        source_image_path = os.path.join(source_folder, image_filename)
        dest_image_path = os.path.join(dest_folder, image_filename)
        if os.path.exists(source_image_path):
            shutil.copy(source_image_path, dest_image_path)
        else:
            print(f"Warning: {source_image_path} does not exist.")

# Copy the images for both training and testing sets
copy_images(train_df, train_folder)
copy_images(test_df, test_folder)

# Save the new TSV files in the root folder
train_df.to_csv('train_labels.tsv', sep='\t', index=False)
test_df.to_csv('test_labels.tsv', sep='\t', index=False)

print("Images have been copied and TSV files created successfully.")
