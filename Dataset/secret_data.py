#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 主要作用是生成secret图片
import sys

sys.path.append('../')
import os
import string
import random
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2 as cv

H, W = 1024, 2048
TEST = True  # linux下改为False


def gen_data(word="", w=W, h=H):
    # 有大字，去除LSB水印
    if word == "":
        word = gen_words()
    pic = w2PIL(word)
    pic = np.array(pic, dtype=np.float32)
    # src2 = pic_reshape(pic).astype('uint8')  # 此处数值类型必须要该，否则会造成int+float32造成失真
    pic2 = cv.resize(pic, (w, h)).astype('uint8')
    # print(w,h)
    src2 = np.where(pic2 > 100, 255, 0)
    src2 = src2[:, :, 0]
    src2 = np.array(src2, np.uint8)
    cv.imwrite("secret.png",src2)
    return src2


def w2PIL(text):
    """
    有字符串生成对应图像
    @param text: secret(string)
    @return:  secret(image)
    """
    im = Image.new("RGB", (140, 63), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    if TEST:
        # windows 下字体文件
        font = ImageFont.truetype(os.path.join("fonts", "simsun.ttc"), 18)  # windows
    else:
        # linux下
        font = ImageFont.truetype(
            r"/home/liu/anaconda3/envs/deeplearn/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/cmss10.ttf",
            18)
    dr.text((0, 0), text, font=font, fill="#000000")
    im = np.array(im, dtype=np.uint8)
    im = np.where(im > 240, 0, 255)
    return im


def pic_reshape(src):
    """
    把文字图片src 反复多次，指导其为桌面分辨率大小.这个虽然现在没有什么用处，但是之后可能会用到。
    使用方法为将W,H 改为较高分辨率
    :param src:   文字图片
    :return:   文字图片  分辨率变大
    """
    srcB = np.zeros((H, W, 3))
    row, col = H, W
    a, b, _ = src.shape
    for i in range(row // a):
        for j in range(col // b):
            srcB[a * i:a * i + a, b * j:b * j + b, :] = src[:, :, :]
    return srcB


def gen_words():
    """
    gen data include ip,name,time
    @return string
    """
    name = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(9))  # 6位字母数字作为用户名
    time = ''.join(random.choice(string.digits) for _ in range(12))  # 12位数字作为时间

    ip = ''.join(random.choice(string.digits) for _ in range(3)) + '.'
    ip = ip + ''.join(random.choice(string.digits) for _ in range(3)) + '.'
    ip = ip + ''.join(random.choice(string.digits) for _ in range(3)) + '.'
    ip = ip + ''.join(random.choice(string.digits) for _ in range(3))  # 4段的3位数字作为Ip地址
    word = name + '\n' + ip + '\n' + time
    return word


def gen_bit_data(secret_size):
    # 生成bit流
    import torch
    secret = np.random.binomial(1, 0.5, secret_size)
    secret = torch.from_numpy(secret).float()
    return secret


def Test_gen_data(word):
    # 有大字，去除LSB水印
    # word = gen_words()
    pic = w2PIL(word)
    pic = np.array(pic, dtype=np.float32)
    # src2 = pic_reshape(pic).astype('uint8')  # 此处数值类型必须要该，否则会造成int+float32造成失真
    pic2 = cv.resize(pic, (W, H))
    src2 = np.where(pic2 > 100, 255, 0)
    src2 = src2[:, :, 0]
    src2 = np.array(src2, np.uint8)
    return src2


if __name__ == "__main__":
    # t2 = gen_data()
    # cv.imshow("t",t2)
    # cv.waitKey(0)
    # cv.imwrite("test.png",t2)
    ss = Test_gen_data("F1h3AlKw5\n445.342.796.448\n939073726462")
    cv.imshow("1", ss)
    cv.waitKey(0)
    cv.imwrite("1.png", ss)
