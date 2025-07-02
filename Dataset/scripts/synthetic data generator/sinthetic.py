from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, List
import random
from tqdm import tqdm
import time


class ImageGenerator:
    def __init__(self, font_dir: str, output_dir: str):
        self.font_dir = Path(font_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.font_cache: Dict[str, ImageFont.FreeTypeFont] = {}
        self.font_paths = list(self.font_dir.glob("*.ttf"))

    def preload_fonts(self, font_size: int = 50) -> None:
        """Preload all fonts into memory to avoid repeated disk access"""
        print("Preloading fonts...")
        for font_path in tqdm(self.font_paths, desc="Loading fonts"):
            self.font_cache[str(font_path)] = ImageFont.truetype(
                str(font_path), font_size
            )

    @staticmethod
    def create_text_image(
        text: str, font: ImageFont.FreeTypeFont, rotation: float, padding: int = 10
    ) -> Image.Image:
        """Create a single text image with optimized processing"""
        text_width, text_height = font.getsize(text)

        image = Image.new(
            "L", (text_width + 2 * padding, text_height + 2 * padding), 255
        )

        draw = ImageDraw.Draw(image)
        draw.text((padding, padding), text, font=font, fill=0)

        return image.point(lambda p: 255 if p < 100 else 0).rotate(
            rotation, expand=True, resample=Image.BILINEAR
        )

    def generate_images(self, prompts: List[str]) -> List[str]:
        """Generate images for all prompts with all fonts"""
        tsv_entries = []
        total_operations = len(prompts) * len(self.font_paths)

        with tqdm(total=total_operations, desc="Generating images") as pbar:
            for i, prompt in enumerate(prompts):
                for j, font_path in enumerate(self.font_paths):
                    font = self.font_cache.get(str(font_path))
                    if not font:
                        pbar.update(1)
                        continue

                    rotation = random.uniform(-15, 15)

                    image = self.create_text_image(prompt, font, rotation)
                    output_path = self.output_dir / f"{i}_{j}.jpg"
                    image.save(str(output_path), optimize=True)

                    tsv_entries.append(f"{output_path.name}\t{prompt}\n")
                    pbar.update(1)

        return tsv_entries


def load_prompts(prompt_file: str) -> List[str]:
    """Load and preprocess prompts"""
    with open(prompt_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    FONT_DIR = "trOCR/Dataset/synthetic data generator/fonts"
    OUTPUT_DIR = "trOCR/Dataset/data/train"
    PROMPT_FILE = "trOCR/Dataset/raw_data/word-level/train-synth.txt"

    generator = ImageGenerator(FONT_DIR, OUTPUT_DIR)
    generator.preload_fonts()

    print("Loading prompts...")
    prompts = load_prompts(PROMPT_FILE)
    print(f"Loaded {len(prompts)} prompts")

    tsv_entries = generator.generate_images(prompts)

    print("Writing TSV file...")
    tsv_path = Path("trOCR/Dataset/data/train.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.writelines(tsv_entries)

    print(f"Results written to {tsv_path}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Execution time: {time.time() - start_time:.2f} seconds")
