import os
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random
import skimage.filters
from skimage.util import random_noise
from tqdm import tqdm
import colorsys


base_dir = "."
num_augmentations = 1


def to_apply():
    return random.choice([True, False])


def noisy(noise_typ, image):
    img_array = np.array(image)

    if noise_typ == "gauss":
        noisy_array = random_noise(img_array, mode="gaussian")
    elif noise_typ == "s&p":
        noisy_array = random_noise(img_array, mode="s&p")
    elif noise_typ == "poisson":
        noisy_array = random_noise(img_array, mode="poisson")

    noisy_array = (noisy_array * 255).astype(np.uint8)

    return Image.fromarray(noisy_array)


def apply_random_enhancements(image):

    rotation = random.randint(-10, 10)
    image = image.rotate(rotation, expand=True, fillcolor=(255, 255, 255))

    if to_apply():
        color_factor = random.uniform(1, 5)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(color_factor)

    if to_apply():
        h = random.uniform(0, 1)
        s = random.uniform(0.43, 1)
        l = random.uniform(0.2, 0.6)
        r, g, b = colorsys.hls_to_rgb(h, l, s)

        r, g, b = int(r * 255), int(g * 255), int(b * 255)

        color_layer = Image.new("RGB", image.size, (r, g, b))

        alpha = random.uniform(0.1, 0.6)
        image = Image.blend(image, color_layer, alpha)

    if to_apply():
        brightness_factor = random.uniform(0.5, 1.5)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)

    if to_apply():
        sharpness_factor = random.uniform(0.5, 1.5)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness_factor)

    if to_apply():
        contrast_factor = random.uniform(0.5, 1.5)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)

    if to_apply():
        blur_factor = random.uniform(0, 2)
        image = image.filter(ImageFilter.GaussianBlur(blur_factor))

    if to_apply():
        noise = random.choice(["gauss", "poisson", "s&p"])
        image = noisy(noise, image)

    return image


def augment_images(folder_name):
    original_folder = os.path.join(base_dir, folder_name)
    augmented_folder = os.path.join(base_dir, f"{folder_name}_augmented")
    original_tsv = os.path.join(f"{folder_name}_labels.tsv")
    augmented_tsv = os.path.join(augmented_folder, f"_labels_augmented.tsv")

    os.makedirs(augmented_folder, exist_ok=True)

    df = pd.read_csv(original_tsv, sep="\t")

    new_entries = []

    with tqdm(total=len(df) * num_augmentations) as pbar:
        for _, row in df.iterrows():
            image_name = row["file"]
            label = row["labels"]
            image_path = os.path.join(original_folder, image_name)

            with Image.open(image_path) as img:
                for i in range(num_augmentations):
                    augmented_img = apply_random_enhancements(img)

                    new_image_name = f"{os.path.splitext(image_name)[0]}_aug_{i+1}.jpg"
                    new_image_path = os.path.join(augmented_folder, new_image_name)

                    augmented_img.save(new_image_path)

                    new_entries.append({"file": new_image_name, "labels": label})
                    pbar.update(1)

    new_df = pd.DataFrame(new_entries)

    updated_df = pd.concat([df, new_df], ignore_index=True)

    updated_df.to_csv(augmented_tsv, sep="\t", index=False)

    print(
        f"Data augmentation completed for {folder_name}. Augmented images and updated label file have been created."
    )


augment_images("C:\\Users\\Dulan\\Code\\pandith\\synth-hw\\test")
augment_images("C:\\Users\\Dulan\\Code\\pandith\\synth-hw\\train")
