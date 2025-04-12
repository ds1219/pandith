import os
import imagehash
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm


def calculate_image_hash(image_path):
    """Calculate perceptual hash of an image."""
    try:
        with Image.open(image_path) as img:
            return str(imagehash.average_hash(img))
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def calculate_white_pixel_percentage(image_path):
    """Calculate percentage of white pixels in an image."""
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale
            img = img.convert("L")
            # Convert to binary (black and white)
            img = img.point(lambda x: 255 if x > 128 else 0, "1")
            # Convert to numpy array
            img_array = np.array(img)
            # Calculate percentage of white pixels (True values in binary image)
            white_percentage = (img_array > 0).sum() / img_array.size * 100
            return white_percentage
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def remove_duplicate_images(directory):
    """Remove images with low white pixel percentage from directory recursively."""
    # Dictionary to store hash: path pairs
    hash_dict = {}
    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
    low_white_pixels = []

    # Scan directory recursively
    for filepath in Path(directory).rglob("*"):
        if filepath.suffix.lower() in image_extensions:
            print(f"Processing: {filepath}")

            # Check white pixel percentage
            white_percentage = calculate_white_pixel_percentage(filepath)
            if white_percentage is not None and white_percentage < 20:
                low_white_pixels.append(str(filepath))
                continue

            # # Duplicate check (commented out)
            # image_hash = calculate_image_hash(filepath)
            # if image_hash:
            #     if image_hash in hash_dict:
            #         duplicates.append(str(filepath))
            #     else:
            #         hash_dict[image_hash] = str(filepath)

    # Remove files
    for file_path in low_white_pixels:  # removed duplicates from list
        try:
            os.remove(file_path)
            print(f"Removed low white pixel image: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")

    print(f"\nSummary:")
    # print(f"Total unique images: {len(hash_dict)}")
    # print(f"Total duplicates removed: {len(duplicates)}")
    print(f"Total low white pixel images removed: {len(low_white_pixels)}")


if __name__ == "__main__":
    # Directory to scan for duplicates
    target_directory = "Dataset/synthetic-handwritten/letters"
    remove_duplicate_images(target_directory)
