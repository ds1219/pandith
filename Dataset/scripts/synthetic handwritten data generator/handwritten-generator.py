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
            # convert array to dictionary with element as key and index as value shifted by 1
            # to make it 1 based index
            data = {item: index + 1 for index, item in enumerate(data["letters"])}
            return data
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def read_image_segments():
    # Use os.path.join for cross-platform path construction
    # and glob to find files within any subfolder of 'letters'
    glob_path = os.path.join("Dataset", "scripts", "synthetic handwritten data generator", "letters", "*", "*.jpg")
    files = glob(glob_path, recursive=True)

    segments_dict = {}
    print(f"Searching for images with glob pattern: {glob_path}")
    print(f"Found {len(files)} image files.")

    if not files:
        print("Warning: No JPG files found matching the glob pattern. Please check your 'Dataset/scripts/synthetic handwritten data generator/letters' directory.")
        return {} # Return an empty dictionary if no files are found

    for file_path in files:
        # Use os.path.split and os.path.basename for robust path parsing
        # file_path example: Dataset\scripts\synthetic handwritten data generator\letters\97\72vld.jpg
        
        # Get the filename with extension (e.g., '72vld.jpg')
        full_filename = os.path.basename(file_path) 
        # Get the filename without extension (e.g., '72vld')
        file_name_without_ext = os.path.splitext(full_filename)[0]

        # Get the directory containing the file (e.g., 'Dataset\...\letters\97')
        parent_dir = os.path.dirname(file_path)
        # Get the name of the immediate parent folder (e.g., '97')
        folder_name = os.path.basename(parent_dir)

        # Debug prints to see what values are being extracted
        # print(f"Processing: {file_path}")
        # print(f"  Extracted folder_name: '{folder_name}'")
        # print(f"  Extracted file_name (without ext): '{file_name_without_ext}'")

        # Ensure folder_name is not empty or malformed before using it as a key
        if not folder_name:
            print(f"Warning: Could not extract a valid folder name from path: {file_path}. Skipping.")
            continue # Skip this file if folder_name is empty

        if folder_name not in segments_dict:
            segments_dict[folder_name] = []
        
        # Append the filename without extension
        segments_dict[folder_name].append(file_name_without_ext)

    print(f"Successfully populated segments_dict with {len(segments_dict)} unique folders.")
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
    # Extract token values
    token_values = greedy_tokenize(input_str, letters)
    return token_values


def concat_images(images):
    """Generate composite of all supplied images in the list, arranged horizontally."""
    if not images:
        return None

    # Add up all the widths
    width = sum(image.width for image in images)
    # Get the highest height
    height = max(image.height for image in images)

    composite = Image.new("RGB", (width, height))
    # Paste each image to the right of the previous one, centered vertically
    x = 0
    for image in images:
        y = (height - image.height) // 2  # Calculate vertical center position
        composite.paste(image, (x, y))
        x += image.width
    return composite


def convert_segments_to_image(segments):
    """Generate a composite image from segment numbers"""
    before_images = []
    after_images = []

    # Separate segments into before and after images
    for segment in segments:
        segment_str = str(segment)
        if segment_str not in image_segments:
            continue

        img_name = random.choice(image_segments[segment_str])
        img_path = f"Dataset/scripts/synthetic handwritten data generator/letters/{segment_str}/{img_name}.jpg"
        img = Image.open(img_path).convert("RGB")

        if segment in [2, 3, 4, 5, 6, 7]:  # Before characters
            before_images.append(img)
        else:
            after_images.append(img)

    # Combine before and after images
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
    rejected_words = []  # List to track rejected words

    for idx, word in tqdm(enumerate(words), total=len(words), desc="Processing words"):
        try:
            segments = convert_word_to_segments(word)
            image = convert_segments_to_image(segments)
            #print(f"{segments} - {image}")

            if image:
                filename = f"{idx}.jpg"
                output_path = os.path.join(output_dir, filename)
                image.save(output_path)
                labels_data.append({"file": filename, "labels": word})
        except Exception:
            rejected_words.append(word)  # Add to rejected words list

    # Save rejected words to file
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

    # Create output directory if it doesn't exist
    output_dir = "output/synthetic-handwritten"
    os.makedirs(output_dir, exist_ok=True)

    # Read words from file
    words = read_words_from_file("Dataset/raw_data/word-level/real.txt")

    # Process words and generate images
    labels_data = process_words(words, output_dir)

    # Save labels to TSV file
    tsv_path = os.path.join(output_dir, "test.tsv")
    save_labels_to_tsv(labels_data, tsv_path)
