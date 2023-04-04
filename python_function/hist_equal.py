import os
import cv2
import numpy
import numpy as np


def main(image_path):
    # 彩色读取
    img = cv2.imread(image_path)
    # convert image to gray without using any library
    img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    print(img.shape)
    # 保存原始灰度化图片
    cv2.imwrite(os.path.join(os.path.dirname(image_path), "gray_" + os.path.basename(image_path)), img)
    equ = cv2.equalizeHist(img)
    res = np.hstack((img, equ))
    # 保存图片
    cv2.imwrite(os.path.join(os.path.dirname(image_path), "hist_" + os.path.basename(image_path)), img)


if __name__ == "__main__":
    main(r"C:\Users\xuyic\Desktop\1 (1015).jpg")
