from PIL import Image, ImageDraw, ImageFont
from glob import glob
import random
from multiprocessing import Pool, Queue, Manager
import os
from functools import partial
import time
from tqdm import tqdm


def get_font_paths():
    return glob("Dataset/synthetic data generator/fonts/*.ttf")


def get_prompts():
    with open("Dataset/train-synth.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def generate_image(args):
    text, font_path, output, font_cache = args
    random_rotation = random.randint(-15, 15)
    font_size = 50

    # Get font from cache or load it
    if font_path not in font_cache:
        font_cache[font_path] = ImageFont.truetype(font_path, font_size)
    font = font_cache[font_path]

    # Measure text size
    text_width, text_height = font.getsize(text)
    image = Image.new("RGBA", (text_width + 20, text_height + 20), (255, 255, 255, 255))

    draw = ImageDraw.Draw(image)
    draw.text((10, 10), text, font=font, fill=(0, 0, 0, 255))

    # Process image
    image = (
        image.convert("L")
        .point(lambda p: 255 if p > 100 else 0)
        .convert("1")
        .point(lambda p: 255 - p)
        .rotate(random_rotation, expand=True)
    )

    # Save image
    image.save(output)
    image.close()

    return f"{os.path.basename(output)}\t{text}\n"


def process_batch(batch_id, batch, font_paths, output_dir, tsv_queue):
    font_cache = {}

    for i, prompt in enumerate(batch):
        random_font = random.choice(font_paths)
        output_path = f"{output_dir}/{batch_id}_{i}.jpg"
        result = generate_image((prompt, random_font, output_path, font_cache))
        tsv_queue.put(result)


def main():
    output_dir = "Dataset/data/synth/train"
    os.makedirs(output_dir, exist_ok=True)

    font_paths = get_font_paths()
    prompts = get_prompts()
    num_processes = os.cpu_count()
    batch_size = len(prompts) // num_processes

    with Manager() as manager:
        tsv_queue = manager.Queue()

        batches = [
            prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)
        ]

        with Pool(processes=num_processes) as pool:
            pool.starmap(
                process_batch,
                [
                    (batch_id, batch, font_paths, output_dir, tsv_queue)
                    for batch_id, batch in enumerate(batches)
                ],
            )

        with open("Dataset/data/synth/train.tsv", "w", encoding="utf-8") as f:
            while not tsv_queue.empty():
                f.write(tsv_queue.get())


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Execution time: {time.time() - start_time:.2f} seconds")
