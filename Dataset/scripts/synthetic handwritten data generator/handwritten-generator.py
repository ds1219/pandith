# read json file
import json
from glob import glob
import random
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm


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
        print(f"Error: File '{file_path}' not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{file_path}'")
        return None


def read_image_segments():
    glob_path = "Dataset/synthetic-handwritten/letters/*/*.jpg"
    files = glob(glob_path, recursive=True)

    segments_dict = {}
    for file_path in files:
        # Get file name without extension
        file_name = file_path.split("\\")[-1].split(".")[0]
        # Get the last folder name
        folder_name = file_path.split("\\")[-2]

        if folder_name not in segments_dict:
            segments_dict[folder_name] = []
        segments_dict[folder_name].append(file_name)

    return segments_dict


def validate_word(word):
    """Check if all characters in word can be mapped to defined letters."""
    i = 0
    while i < len(word):
        found = False
        for length in range(min(4, len(word) - i), 0, -1):
            if word[i : i + length] in letters:
                i += length
                found = True
                break
        if not found:
            return False
    return True


def convert_word_to_segments(word):
    if not validate_word(word):
        return None

    segments = []
    i = 0
    while i < len(word):
        found = False
        # Try longest possible matches first
        for length in range(min(4, len(word) - i), 0, -1):
            substr = word[i : i + length]
            if substr in letters:
                segments.append(letters[substr])
                i += length
                found = True
                break

        if not found:
            i += 1
    return segments


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


def is_before_character(segment_num):
    """Check if the segment number corresponds to a character that should go before."""
    # Add the segment numbers that represent "before" characters
    before_segments = [2, 3, 4, 5, 6, 7]  # Adjust these numbers based on your mapping
    return segment_num in before_segments


def convert_segments_to_image(segments):
    images = []
    before_images = []
    after_images = []

    # First, separate the segments into before and after images
    for i, segment in enumerate(segments):
        img_name = random.choice(image_segments[str(segment)])
        img_path = f"Dataset/synthetic-handwritten/letters/{segment}/{img_name}.jpg"
        img = Image.open(img_path).convert("RGB")

        if is_before_character(segment):
            before_images.append(img)
        else:
            after_images.append(img)

    # Combine the lists with before images first
    images = before_images + after_images
    return concat_images(images)


def read_words_from_file(file_path):
    """Read words from a text file, one word per line."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"Error: Words file '{file_path}' not found")
        return []


def save_labels_to_tsv(labels_data, output_path):
    """Save image filenames and their labels to a TSV file."""
    df = pd.DataFrame(labels_data, columns=["file", "labels"])
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Labels saved to: {output_path}")


if __name__ == "__main__":
    letters = read_json_file("Dataset/synthetic-handwritten/letters.json")
    image_segments = read_image_segments()

    # Create output directory if it doesn't exist
    output_dir = "output/synthetic-handwritten"
    os.makedirs(output_dir, exist_ok=True)

    # Read words from file
    words = read_words_from_file("Dataset/raw_data/word-level/test-real.txt")

    # List to store filename and label pairs
    labels_data = []

    for word in tqdm(words, desc="Generating images", unit="word"):
        result = convert_word_to_segments(word)
        if result is None:
            print(f"Skipping '{word}': contains undefined characters")
            continue

        # Generate and save image
        image = convert_segments_to_image(result)
        if image:
            filename = f"{word}.jpg"
            output_path = os.path.join(output_dir, filename)
            image.save(output_path)
            print(f"Generated image for '{word}': {output_path}")
            print(f"Segments: {result}")

            # Add to labels data
            labels_data.append({"file": filename, "labels": word})

    # Save labels to TSV file
    tsv_path = os.path.join(output_dir, "test.tsv")
    save_labels_to_tsv(labels_data, tsv_path)
