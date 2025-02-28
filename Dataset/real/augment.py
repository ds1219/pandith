import os
import pandas as pd
from PIL import Image, ImageEnhance
import random

# Define file paths and parameters
base_dir = '.'  # Base directory containing train and test folders
num_augmentations = 25  # Number of augmented copies per image

# Function to apply random enhancements to an image
def apply_random_enhancements(image):
    # Randomly adjust rotation make sure to adjust the image size
    rotation = random.randint(-10, 10)
    image = image.rotate(rotation, expand=True)
    
    return image

# Function to augment images in a given folder
def augment_images(folder_name):
    # Paths for original and augmented data
    original_folder = os.path.join(base_dir, folder_name)
    augmented_folder = os.path.join(base_dir, f"{folder_name}_augmented")
    original_tsv = os.path.join(f"{folder_name}_labels.tsv")
    augmented_tsv = os.path.join(augmented_folder, f"_labels_augmented.tsv")

    # Create augmented folder if it doesn't exist
    os.makedirs(augmented_folder, exist_ok=True)

    # Read the original TSV file into a DataFrame
    df = pd.read_csv(original_tsv, sep='\t')

    # List to store new label entries
    new_entries = []

    # Process each image in the DataFrame
    for _, row in df.iterrows():
        image_name = row['file']
        label = row['labels']
        image_path = os.path.join(original_folder, image_name)

        # Open the original image
        with Image.open(image_path) as img:
            # Generate augmented images
            for i in range(num_augmentations):
                # Apply random enhancements
                augmented_img = apply_random_enhancements(img)

                # Create a new filename for the augmented image
                new_image_name = f"{os.path.splitext(image_name)[0]}_aug_{i+1}.jpg"
                new_image_path = os.path.join(augmented_folder, new_image_name)

                # Save the augmented image
                augmented_img.save(new_image_path)

                # Append the new entry to the list
                new_entries.append({'ile': new_image_name, 'labels': label})

    # Create a DataFrame for the new entries
    new_df = pd.DataFrame(new_entries)

    # Concatenate the original DataFrame with the new entries
    updated_df = pd.concat([df, new_df], ignore_index=True)

    # Save the updated DataFrame to a new TSV file
    updated_df.to_csv(augmented_tsv, sep='\t', index=False)

    print(f"Data augmentation completed for {folder_name}. Augmented images and updated label file have been created.")

# Augment images in both train and test folders
augment_images('Dataset/real/test')
augment_images('Dataset/real/train')
