from matplotlib import pyplot as plt
import numpy as np
import cv2

def show_cv2_image(image):
    plt.imshow(image, cmap="gray")
    plt.show()

def otsu_threshold(img):
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # ostu threshold
    _, ostu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ostu

def split_words(img):
    kernel = np.ones((3, 15), np.uint8)
    dilated2 = cv2.dilate(img, kernel, iterations=5)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        dilated2, connectivity=8
    )
    words = []

    # Loop through each connected component
    for i in range(1, num_labels):  # Start from 1 to skip the background
        # Extract bounding box of the component

        # sort stats by x coordinate
        stats = stats[stats[:, 0].argsort()]

        x, y, w, h, area = stats[i]

        # Ignore small components based on area if necessary
        if area < 10000:  # You can adjust the threshold
            continue

        # Extract the component
        component = img[y : y + h, x : x + w]

        words.append(component)

    return words


def preprocess(img):
    img = otsu_threshold(img)
    img = cv2.bitwise_not(img)
    return img



# read images in folder
import glob
import os

path = "Dataset/real/processed/"
save_path = "Dataset/real/preprocessed"

for filename in glob.glob(os.path.join(path, "*.jpg")):
    print(filename)
    img = cv2.imread(filename)
    img = preprocess(img)
    cv2.imwrite(os.path.join(save_path, os.path.basename(filename)), img)
    
