# read json file
import json
from glob import glob
import random
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
import re


def read_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            data.get("letters", [])
            data = {item: index + 1 for index, item in enumerate(data["letters"])}
            return data
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def read_image_segments():
    glob_path = os.path.join(
        "Dataset",
        "scripts",
        "synthetic handwritten data generator",
        "letters",
        "*",
        "*.jpg",
    )
    files = glob(glob_path, recursive=True)

    segments_dict = {}
    print(f"Searching for images with glob pattern: {glob_path}")
    print(f"Found {len(files)} image files.")

    if not files:
        print(
            "Warning: No JPG files found matching the glob pattern. Please check your 'Dataset/scripts/synthetic handwritten data generator/letters' directory."
        )
        return {}
    for file_path in files:

        full_filename = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(full_filename)[0]

        parent_dir = os.path.dirname(file_path)
        folder_name = os.path.basename(parent_dir)
        if not folder_name:
            print(
                f"Warning: Could not extract a valid folder name from path: {file_path}. Skipping."
            )
            continue

        if folder_name not in segments_dict:
            segments_dict[folder_name] = []

        segments_dict[folder_name].append(file_name_without_ext)

    print(
        f"Successfully populated segments_dict with {len(segments_dict)} unique folders."
    )
    return segments_dict


def greedy_tokenize(s, token_dict):
    need_to_split = {"ේ": "ේ", "ො": "ො", "ෝ": "ො◌්", "ෞ": "ෞ", "◌": ""}
    for key, value in need_to_split.items():
        s = s.replace(key, value)

    for i in range(0, len(s)):
        if s[i] == "ෙ" or s[i] == "ෛ":
            s = s.replace(f"{s[i-1]}{s[i]}", f"{s[i]}{s[i-1]}")
    tokens = []
    i = 0
    while i < len(s):
        match = None
        for j in range(len(s), i, -1):
            if s[i:j] in token_dict:
                match = s[i:j]
                break
        if match:
            tokens.append(token_dict[match])
            i += len(match)
        else:
            raise ValueError(f"Unrecognized token at index {i}: {s[i]}")
    return tokens


def convert_word_to_segments(input_str):
    token_values = greedy_tokenize(input_str, letters)
    return token_values


def concat_images(images):
    """Generate composite of all supplied images in the list, arranged horizontally."""
    if not images:
        return None

    width = sum(image.width for image in images)
    height = max(image.height for image in images)

    composite = Image.new("RGB", (width, height))
    x = 0
    for image in images:
        y = (height - image.height) // 2
        composite.paste(image, (x, y))
        x += image.width
    return composite


def convert_segments_to_image(segments):
    """Generate a composite image from segment numbers"""
    before_images = []
    after_images = []

    for segment in segments:
        segment_str = str(segment)
        if segment_str not in image_segments:
            continue

        img_name = random.choice(image_segments[segment_str])
        img_path = f"Dataset/scripts/synthetic handwritten data generator/letters/{segment_str}/{img_name}.jpg"
        img = Image.open(img_path).convert("RGB")

        if segment in [2, 3, 4, 5, 6, 7]:
            before_images.append(img)
        else:
            after_images.append(img)

    return concat_images(before_images + after_images)


def read_words_from_file(file_path):
    """Read words from a text file, one word per line."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        return []


def save_labels_to_tsv(labels_data, output_path):
    """Save image filenames and their labels to a TSV file."""
    df = pd.DataFrame(labels_data, columns=["file", "labels"])
    df.to_csv(output_path, sep="\t", index=False)


def process_words(words, output_dir):
    """Process a list of words and generate handwritten images"""
    labels_data = []
    rejected_words = []

    for idx, word in tqdm(enumerate(words), total=len(words), desc="Processing words"):
        try:
            segments = convert_word_to_segments(word)
            image = convert_segments_to_image(segments)
            # print(f"{segments} - {image}")

            if image:
                filename = f"{idx}.jpg"
                output_path = os.path.join(output_dir, filename)
                image.save(output_path)
                labels_data.append({"file": filename, "labels": word})
        except Exception:
            rejected_words.append(word)

    if rejected_words:
        rejected_path = os.path.join(output_dir, "rejected.txt")
        with open(rejected_path, "w", encoding="utf-8") as f:
            f.write("\n".join(rejected_words))

    return labels_data


if __name__ == "__main__":
    letters = read_json_file(
        "Dataset/scripts/synthetic handwritten data generator/letters.json"
    )

    image_segments = read_image_segments()

    output_dir = "output/synthetic-handwritten"
    os.makedirs(output_dir, exist_ok=True)

    words = read_words_from_file("Dataset/raw_data/word-level/real.txt")

    labels_data = process_words(words, output_dir)

    tsv_path = os.path.join(output_dir, "test.tsv")
    save_labels_to_tsv(labels_data, tsv_path)
