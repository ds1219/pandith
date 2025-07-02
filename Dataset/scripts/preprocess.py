from matplotlib import pyplot as plt
import numpy as np
import cv2


def show_cv2_image(image):
    plt.imshow(image, cmap="gray")
    plt.show()


def otsu_threshold(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, ostu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ostu


def preprocess(img):
    img_medium = remove_shadows(img)
    img_medium = otsu_threshold(img)

    for i in img_medium:
        show_cv2_image(i)
        otsu_threshold(i)

        cv2.waitKey(0)
        pass
    return img


def remove_shadows(img):
    # https://stackoverflow.com/a/44752405
    rgb_planes = cv2.split(img)

    result_planes = []
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
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)

    return result_norm


# read images in folder
import glob
import os

path = "C:/Users/Dulan/Desktop/New folder"
save_path = "C:/Users/Dulan/Desktop/New folder (2)"

for filename in glob.glob(os.path.join(path, "*.jpg")):
    print(filename)
    img = cv2.imread(filename)
    img = preprocess(img)
    cv2.imwrite(os.path.join(save_path, os.path.basename(filename)), img)
