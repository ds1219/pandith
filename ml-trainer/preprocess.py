from matplotlib import pyplot as plt
import numpy as np
import cv2


def show_cv2_image(image):
    plt.imshow(image, cmap="gray")
    plt.show()


def de_noise_image(image):
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)


def remove_shadows(img):
    # https://stackoverflow.com/questions/44752240/how-to-remove-shadow-from-scanned-images-using-opencv
    rgb_planes = cv2.split(img)

    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(
            diff_img,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8UC1,
        )
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_norm_planes)

    return result


def remove_page_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Remove horizontal lines
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 81, 17
    )
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))

    # Using morph close to get lines outside the drawing
    remove_horizontal = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=3
    )
    cnts = cv2.findContours(
        remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    mask = np.zeros(gray.shape, np.uint8)
    for c in cnts:
        cv2.drawContours(mask, [c], -1, (255, 255, 255), 2)

    # First inpaint
    img_dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    gray_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_dst, 50, 150, apertureSize=3)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))

    # Using morph open to get lines inside the drawing
    opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    mask = np.uint8(img_dst)
    mask = np.zeros(gray_dst.shape, np.uint8)
    for c in cnts:
        cv2.drawContours(mask, [c], -1, (255, 255, 255), 2)

    # Second inpaint
    return cv2.inpaint(img_dst, mask, 3, cv2.INPAINT_TELEA)


def binarize_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


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
    img = de_noise_image(img)
    show_cv2_image(img)
    img = remove_shadows(img)
    show_cv2_image(img)
    img = remove_page_lines(img)
    show_cv2_image(img)
    img = binarize_image(img)
    show_cv2_image(img)
    img = cv2.resize(img, (int(500 * img.shape[1] / img.shape[0]), 500))
    img = cv2.bitwise_not(img)
    show_cv2_image(img)
    img = split_words(img)
    return img


img = cv2.imread(
    "C:/Users/Dulan/Documents/Github/pandithaya/Dataset/data/train-real/test/22-7.jpg"
)


img = preprocess(img)
