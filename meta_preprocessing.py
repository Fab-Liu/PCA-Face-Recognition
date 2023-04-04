######## 图像预处理元函数合集 ########
import cv2
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


######## 基本操作 ########
## 图像导入 ##
def load_img(fname):
    img = cv2.imread(fname, cv2.IMREAD_COLOR)
    return img


## 获取图像宽高信息 ##
def getImgInfo(img):
    w = img.shape[0]
    h = img.shape[1]
    print('Image Size: ', w, '*', h)
    return


## 打印图像 ##
def getImage(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)


######## 图像校正 ########
#### 等比缩放至宽度为W的图像 ####
def Scaler(img, W):
    w = img.shape[0]
    h = img.shape[1]
    k = 1
    if w == 150:
        out = img
    else:
        k = W / w
        w0 = k * h
        h0 = k * w
        out = cv2.resize(img, (int(w0), int(h0)))

    return out


#### 色彩处理 ####
## 灰度化 ##
def toGray(img):
    out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return out


#### 直方图均衡 ####
## 打印直方图 ##
def displayHist(img):
    plt.figure()
    plt.hist(img.ravel(), 256)
    plt.show()


## 直方图均衡化 ##
def equalHist(img):
    out = cv2.equalizeHist(img)
    return out


## 归一化 ##
def normHist(img, a=0, b=255):
    c = img.min()
    d = img.max()

    out = img.copy()
    out = (b - a) / (d - c) * (out - c) + a
    out[out < a] = a
    out[out > b] = b

    out = out.astype(np.uint8)

    return out


######## 滤波 ########
#### 高斯滤波及调参 ####
## 高斯滤波函数 ##
def gsBlur(img, kX, kY):
    out = cv2.GaussianBlur(img, (kX, kY))
    return out


## 调参函数 ##
# 超参数记录：
# 对于 dst = cv2.GaussianBlur(img, (kX, kY))
# (kX, kY)为高斯内核大小，且必须为正数和奇数，本调参函数默认预设从(1,1)开始增长。
# 调参返回值为一个结果集，包含测试中得到的所有结果。
def gsTuning(img, kX, kY):
    rs = []
    if kX < 1 or kY < 1 or (kX % 2 == 0) or (kY % 2 == 0):
        print('Invalid hyperparameter!')
    else:
        for i in range(1, kX + 1, 2):
            for j in range(1, kY, 2):
                dst = cv2.GaussianBlur(img, (i, j))
                rs.append(dst)
        return rs


#### 双边滤波及调参 ####
## 双边滤波函数 ##
def bilaFilter(img, d, sigma):
    out = cv2.bilateralFilter(img, d, sigma)
    return out


## 调参函数 ##
# 超参数记录：
# 对于 dst = cv2.bilateralFilter(img, d, sigmaColor,
#                               sigmaSpace[, dst[, borderType]])
# d: 过滤使用的像素邻域直径，常见取值分界线 d=5，但对于需要大量噪声过滤的离线应用建议选择d=9。
# sigmaColor：颜色空间过滤指标，越大则像素邻域内更远的颜色会被混合产生更大的半等色区域。
# sigmaSpace: 坐标空间过滤指标，越大则更远的颜色相近的像素会互相影响。若d>0，则sigmaSpace参数不影响结果。
# 对于sigma值的设定为0到255，但当sigma>150时，照片的卡通化效果就会非常强烈。
# 调参函数需要填入非负的d值，并以1为步长增加；
# 以及0-180之间（因为再增加意义不太大且算力需求特别高）的sigma最大值（最好为10的倍数），并以10为步长增长。
# 调参返回值为一个结果集，包含测试中得到的所有结果。
def bilaTuning(img, dmax, smax):
    rs = []
    if dmax < 0 or smax <= 0:
        print('Invalid hyperparameter!')
    else:
        for i in range(0, dmax + 1):
            for j in range(0, smax + 1, 10):
                dst = cv2.bilateralFilter(img, i, j)
                rs.append(dst)

        return rs


######## 图片分割与输出结果 ########
#### 阈值化 ####
def getThresh(img):
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


#### 获取分界线 ####
def imageSegment(thresh):
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


#### 展示分界线 ####
def displaySegment(img, contours):
    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
    cv2.imshow('image', img)
    cv2.waitKey(0)
