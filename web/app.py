from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    jsonify,
)
import os
from time import sleep
import threading
import uuid
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer
import torch
from PIL import Image
from secrets import token_hex
import os
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import random
import skimage.filters
from skimage.util import random_noise
from tqdm import tqdm
import colorsys
import cv2
from skimage.filters import threshold_sauvola
from skimage import img_as_ubyte

from huggingface_hub import login


access_token = os.getenv("RUNPOD_SECRET_hf_token")
login = login(token=access_token)
print("logged in")

model_details = "ds1219/Pandithaya-Printed-Words-128"
processor = TrOCRProcessor.from_pretrained(
    model_details, use_fast=True, size={"height": 128, "width": 128}
)
print("downloaded processor")
tokenizer = AutoTokenizer.from_pretrained(model_details)
print("downloaded tokenizer")
processor.tokenizer = tokenizer
model = VisionEncoderDecoderModel.from_pretrained("ds1219/pandith-synth-real")
print("downloaded model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

import sqlite3


def binarize_image(image):

    image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    window_size = 25
    sauvola_thresh = threshold_sauvola(gray, window_size=window_size)
    binary = (gray > sauvola_thresh).astype(np.uint8) * 255
    binary_rgb = cv2.cvtColor(
        binary,
        cv2.COLOR_GRAY2RGB,
    )
    return binary_rgb


def preprocess(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = binarize_image(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img


# Database setup function
def get_db_connection():
    conn = sqlite3.connect("example.db")
    conn.row_factory = sqlite3.Row
    # Create table if it doesn't exist
    cursor = conn.cursor()
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS history (
        img TEXT PRIMARY KEY NOT NULL,
        txt TEXT NOT NULL
    )
    """
    )
    conn.commit()
    return conn


def ocr(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image)
    pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(model.device)
    generated_ids = model.generate(pixel_values, interpolate_pos_encoding=True)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def do_ocr(filename=None):
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename) if filename else None
    text = ocr(filepath)

    # Create a new connection for this thread
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO history (img, txt) VALUES (?, ?)", (filename, text))
    conn.commit()
    conn.close()

    print(text)
    return text


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

processing_tasks = {}


def process_in_background(task_id, filename):
    processing_tasks[task_id] = {"status": "processing", "result": None}
    try:
        result = do_ocr(filename)  # This also saves to history table
        processing_tasks[task_id] = {"status": "complete", "result": result}
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        processing_tasks[task_id] = {"status": "error", "result": str(e)}


@app.route("/")
def index():
    # Fetch history from the database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT img, txt FROM history ORDER BY rowid DESC LIMIT 10")
    history = cursor.fetchall()
    conn.close()

    # Convert history rows to a list of dictionaries for the template
    history_data = [{"img": row["img"], "txt": row["txt"]} for row in history]

    return render_template("upload.html", history=history_data)


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part", 400
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400
        if file and file.filename.lower().endswith(".jpg"):
            new_name = str(token_hex(16)) + ".jpg"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], new_name)
            file.save(filepath)

            task_id = str(uuid.uuid4())
            processing_thread = threading.Thread(
                target=process_in_background, args=(task_id, new_name)
            )
            processing_thread.start()

            return redirect(
                url_for("processing", task_id=task_id, filename=file.filename)
            )
        else:
            return "Invalid file type. Only .jpg allowed.", 400

    # Get history for the template
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT img, txt FROM history ORDER BY rowid DESC LIMIT 10")
    history = cursor.fetchall()
    conn.close()

    # Convert history rows to a list of dictionaries for the template
    history_data = [{"img": row["img"], "txt": row["txt"]} for row in history]

    return render_template("upload.html", history=history_data)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/processing")
def processing():
    task_id = request.args.get("task_id")
    filename = request.args.get("filename")
    return render_template("processing.html", task_id=task_id, filename=filename)


@app.route("/task_status/<task_id>")
def task_status(task_id):
    if task_id not in processing_tasks:
        return jsonify({"status": "not_found"})
    return jsonify(processing_tasks[task_id])


@app.route("/output")
def output():
    filename = request.args.get("filename")
    task_id = request.args.get("task_id")

    if task_id and task_id in processing_tasks:
        result = processing_tasks[task_id]["result"]
    else:
        result = do_ocr(filename)

    return render_template("output.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
