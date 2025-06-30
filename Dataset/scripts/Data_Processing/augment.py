import os
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random
import skimage.filters
from skimage.util import random_noise
from tqdm import tqdm
import colorsys


# Define file paths and parameters
base_dir = "."  # Base directory containing train and test folders
num_augmentations = 1  # Number of augmented copies per image


def to_apply():
    return random.choice([True, False])


def noisy(noise_typ, image):
    # Convert PIL image to numpy array
    img_array = np.array(image)

    if noise_typ == "gauss":
        noisy_array = random_noise(img_array, mode="gaussian")
    elif noise_typ == "s&p":
        noisy_array = random_noise(img_array, mode="s&p")
    elif noise_typ == "poisson":
        noisy_array = random_noise(img_array, mode="poisson")

    # random_noise returns float values in [0, 1], convert to uint8 for PIL
    noisy_array = (noisy_array * 255).astype(np.uint8)

    # Convert back to PIL image and return
    return Image.fromarray(noisy_array)


# Function to apply random enhancements to an image
def apply_random_enhancements(image):

    # Randomly adjust rotation make sure to adjust the image size
    rotation = random.randint(-10, 10)
    image = image.rotate(rotation, expand=True, fillcolor=(255, 255, 255))

    if to_apply():
        # Randomly adjust color balance
        color_factor = random.uniform(1, 5)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(color_factor)

    if to_apply():
        # Apply a randomly colored filter using HSL
        h = random.uniform(0, 1)  # Hue: 0 to 1
        s = random.uniform(0.43, 1)  # Saturation: 50% to 100%
        l = random.uniform(0.2, 0.6)  # Lightness: 50% to 80%

        # Convert HSL to RGB (values will be in range 0-1)
        r, g, b = colorsys.hls_to_rgb(h, l, s)

        # Scale RGB values to 0-255
        r, g, b = int(r * 255), int(g * 255), int(b * 255)

        # Create a color layer with the HSL-based color
        color_layer = Image.new("RGB", image.size, (r, g, b))

        # Random transparency for the color filter
        alpha = random.uniform(0.1, 0.6)  # Subtle effect with alpha between 10% and 50%
        image = Image.blend(image, color_layer, alpha)

    if to_apply():
        # Randomly adjust brightness
        brightness_factor = random.uniform(0.5, 1.5)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)

    if to_apply():
        # randomly adjust sharpness
        sharpness_factor = random.uniform(0.5, 1.5)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness_factor)

    if to_apply():
        # Randomly adjust contrast
        contrast_factor = random.uniform(0.5, 1.5)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)

    if to_apply():
        # randomly add blur
        blur_factor = random.uniform(0, 2)
        image = image.filter(ImageFilter.GaussianBlur(blur_factor))

    if to_apply():
        noise = random.choice(["gauss", "poisson", "s&p"])
        image = noisy(noise, image)

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
    df = pd.read_csv(original_tsv, sep="\t")

    # List to store new label entries
    new_entries = []

    with tqdm(total=len(df) * num_augmentations) as pbar:
        # Process each image in the DataFrame
        for _, row in df.iterrows():
            image_name = row["file"]
            label = row["labels"]
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
                    new_entries.append({"file": new_image_name, "labels": label})
                    pbar.update(1)

    # Create a DataFrame for the new entries
    new_df = pd.DataFrame(new_entries)

    # Concatenate the original DataFrame with the new entries
    updated_df = pd.concat([df, new_df], ignore_index=True)

    # Save the updated DataFrame to a new TSV file
    updated_df.to_csv(augmented_tsv, sep="\t", index=False)

    print(
        f"Data augmentation completed for {folder_name}. Augmented images and updated label file have been created."
    )


# Augment images in both train and test folders
augment_images("C:\\Users\\Dulan\\Code\\pandith\\synth-hw\\test")
augment_images("C:\\Users\\Dulan\\Code\\pandith\\synth-hw\\train")
