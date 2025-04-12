from glob import glob
import cv2
import os
from tqdm import tqdm

image_path = "C:/Users/Dulan/Desktop/archive (2)/Dataset454/source"
save_path = "C:/Users/Dulan/Desktop/archive (2)/Dataset454/cropped"

images = glob(image_path + "/**/*.jpg", recursive=True)
print("Found images: ", len(images))

images = images[78959:-1]
print(images[0])

for i in tqdm(images):
    try:
        img = cv2.imread(i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # denoise the image
        img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # resize the image to 224x224
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        x, y, w, h = cv2.boundingRect(img)
        img = img[y : y + h, x : x + w]
        # save the image
        savefolder = save_path + "/" + i.split("\\")[-2]
        # create the folder if it does not exist
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
        save = savefolder + "/" + i.split("\\")[-1]
        cv2.imwrite(save, img)
    except Exception as e:
        print("Error: ", e)
        print("Image: ", i)
        continue

cv2.destroyAllWindows()
