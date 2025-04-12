import cv2

# read images in folder
import glob
import os

path = "Dataset/archive (1)/train"
save_path = "Dataset/archive (1)/train-inverted"

for filename in glob.glob(os.path.join(path, "*.jpg")):
    print(filename)
    img = cv2.imread(filename)
    img = cv2.bitwise_not(img)
    cv2.imwrite(os.path.join(save_path, os.path.basename(filename)), img)
