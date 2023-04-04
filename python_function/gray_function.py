import os
import cv2

folder = '..\image_train'

for category in os.listdir(folder):
    if category != ".DS_Store":
        for image in os.listdir(os.path.join(folder, category)):
            if image!= ".DS_Store":
                img = cv2.imread(os.path.join(folder, category, image))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # resize to 100x 100
                gray = cv2.resize(gray, (100, 100))
                cv2.imwrite(os.path.join(folder, category, image), gray)