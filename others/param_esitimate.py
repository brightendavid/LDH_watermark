"""
获取拍摄图像和模拟拍摄图像的差别，和扰动参数选择
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from train.train_paper import noise_layer

def mean(src):
    a, b = src.shape
    win_mean = np.sum(np.sum(src, axis=-1), axis=-1) / a / b
    return win_mean


def var(src):
    """
    灰度图像方差为模糊程度
    @param src:
    @return:
    """
    a, b = src.shape
    win_mean = mean(src)
    win_sqr_mean = np.sum(np.sum((src - win_mean) ** 2, axis=-1), axis=-1) / a / b
    # win_var = win_sqr_mean - win_mean ** 2
    return win_sqr_mean

def blur_degree(src1,src2):
    # https://zhuanlan.zhihu.com/p/205495107  模糊度计算
    a = var(src1)
    b = var(src2)
    return a/b

def brightness(src):
    R,G,B = src[0],src[1],src[2]
    bright = 0.299 * R + 0.587 * G + 0.114 * B
    mean_bri = mean(bright)
    return mean_bri


def brighten_degree(src1,src2):
    a = brightness(src1)
    b = brightness(src2)
    return a/b

def jpeg_degree(src1,src2):
    # 查图像属性
    pass


def warp_matrix(src,src2):
    # 图像矫正的扭曲矩阵的逆矩阵，查看对应位置的参数，不需要写这个，扭曲的角度和参数合理即可
    # https://blog.csdn.net/wsp_1138886114/article/details/83374333
    img = cv.imread('1.png')
    row, col = img.shape[:2]
    col = col - 1
    row = row - 1

    a = np.float32([[0, 0], [row, 0], [0, col], [row, col]])
    b = np.float32([[0, 0], [row // 2, 0], [0, col // 2], [row // 2, col // 2]])

    plt.imshow(img)
    plt.show()

    m = cv.getPerspectiveTransform(a, b)
    # print(m)
    # img1 = cv.warpPerspective(img, m, (col // 2, row // 2), borderValue=(255, 255, 255))
    # plt.imshow(img1)
    # plt.show()

def gen_noise_src(src):
    # 通过noise layer获取模拟的src
    src2,_ = noise_layer(src,src)
    return src2


if __name__ == '__main__':
    src = cv.imread(r"C:\Users\brighten\Desktop\new\result\2+3_result\1_15.png", 0)
    src2 = cv.imread(r"C:\Users\brighten\Desktop\new\data2\1.png", 0)
    print(blur_degree(src2,src))
