#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
2022/4/10
测试数据集中的图像
in fact ,in a work for hide or reveal ,we only need to load one model_udh ,rather than load all modles.
"""
import os
import sys

from torchvision import transforms

# from LSB.lsb_data import add_lsb

sys.path.append('../')
import cv2 as cv
import torch.nn as nn
import torch.utils.data.dataloader
from Functions.loss_functions import *
from model_udh import Revealnet, Hide
import numpy as np

model_path1 = r'../save_model/0710/BEST_Encode_07-10watermark_checkpoint49dict.pth'
model_path2 = r'../save_model/0710/BEST_Rev_07-10watermark_checkpoint49dict.pth'


def Reveal(data,is_cuda=False):
    # 大约可以使用2000*2000分辨率的图像进行水印提取
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    checkpoint2 = torch.load(model_path2, map_location=device)
    torch.cuda.empty_cache()
    Rnet = Revealnet.RevealNet(input_nc=3,
                               output_nc=1, norm_layer=nn.BatchNorm2d,
                               output_function=nn.Sigmoid)
    if is_cuda:
        Rnet.cuda()
    else:
        Rnet.cpu()

    Rnet.load_state_dict(checkpoint2['state_dict'])
    Rnet.eval()
    out_sec = Rnet(data)
    out_image = tensor2np(out_sec)
    return out_image


def Hide_secret(data, secret):
    torch.cuda.empty_cache()

    checkpoint1 = torch.load(model_path1, map_location=torch.device("cpu"))
    torch.cuda.empty_cache()
    Hnet = Hide.UnetGenerator(input_nc=1,
                              output_nc=3 * 1, num_downs=5, norm_layer=nn.BatchNorm2d,
                              output_function=nn.Sigmoid)

    Hnet.cpu()

    Hnet.load_state_dict(checkpoint1['state_dict'])
    Hnet.eval()

    data, secret = add_lsb(data, secret)
    data = transforms.Compose([
        transforms.ToTensor()
    ])(data)  # 张量化
    data = data[np.newaxis, :, :, :]
    data = data.type(torch.FloatTensor)

    secret = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])(secret)  # 张量化
    secret = secret[np.newaxis, :, :, :]
    print(secret.shape)

    secret = secret.type(torch.FloatTensor)

    # data = torch.cat((secret, data), 1)
    data = data.type(torch.FloatTensor)
    print(data.shape)

    res = Hnet(secret)
    out_contain = res + data
    out_image = tensor2np(out_contain)
    return out_image


def tensor2np(src):
    output = src.squeeze(0)
    output = np.array(output.cpu().detach().numpy(), dtype='float32')
    output = np.transpose(output, (1, 2, 0))
    output *= 255.0
    return output


def Reveal_one_pic(src,is_cuda=False):
    src = transforms.Compose([
        transforms.ToTensor()
    ])(src)  # 张量化
    src = src[np.newaxis, :, :, :]
    if is_cuda:
        src = src.type(torch.cuda.FloatTensor)
    else:
        src = src.type(torch.FloatTensor)
    print(src.shape)
    out_sectret = Reveal(src,is_cuda=is_cuda)
    cv.imwrite("test_picture/1.png", out_sectret)
    return out_sectret


class Secret_message:
    def __init__(self):
        self.ip = self.get_ip()
        self.time = self.get_time()
        self.name = self.get_name()
        self.secret = self.get_secret()

    def get_ip(self):
        # 能用
        # ip = 169.254.240.224
        import socket
        # print(socket.gethostbyname(socket.getfqdn(socket.gethostname())))
        return str(socket.gethostbyname(socket.getfqdn(socket.gethostname())))

    def get_time(self):
        import datetime
        Time = datetime.datetime.now()
        # print("当前的日期和时间是 %s" % Time)
        # print("当前的年份是 %s" % Time.year)
        # print("当前的月份是 %s" % Time.month)

        year = str(Time.year)
        month = self.format_time(Time.month)
        day = self.format_time(Time.day)
        hour = self.format_time(Time.hour)
        minute = self.format_time(Time.minute)
        second = self.format_time(Time.second)

        # print("当前的日期是  %s" % Time.day)
        # print("当前小时是 %s" % Time.hour)
        # print("当前分钟是 %s" % Time.minute)
        # print("当前秒是  %s" % Time.second)
        Now_time = year + month + day + hour + minute + second
        return Now_time

    def format_time(self, ss):
        if len(str(ss)) < 2:
            ss = '0' + str(ss)
        else:
            ss = str(ss)
        return ss

    def get_name(self):
        getlogin_X = os.getlogin()
        return getlogin_X

    def get_secret(self):
        self.secret = self.name + '-' + self.ip + '-' + self.time
        return self.secret


def test_hide_model():
    torch.cuda.empty_cache()
    path = r"test_picture/f1.png"
    data = cv.imread(path)
    data = data[0:1024, 0:1024]
    secret = "brighten-192.131.324.222-213123213"
    src = Hide_secret(data, secret)
    return src


def Hide_pic_port(data):
    """
    input:data   a numpy image  which has a size most modded by 16
    output:container a numpy image   which has the same size as the input_data
    """
    # data = data[0:1024, 0:1024]
    # data =255-data
    S = Secret_message()
    secret = S.secret # 实时读取数据，主要为切换时间
    # secret = "AFAtsdn-222.232.324.222-213123213"
    torch.cuda.empty_cache()
    src = Hide_secret(data, secret)
    return src


if __name__ == '__main__':
    # S = Secret_message()
    # print(S.secret)
    path = r"./test_picture/2022-04-17 185111.png"
    src = cv.imread(path)
    src = src[0:512, 0:512]
    # Reveal_one_pic(src)
    src = Hide_pic_port(src)
    cv.imwrite("test_picture/2.png", src)

    path = r"test_picture/2.png"
    src = cv.imread(path)
    reve = Reveal_one_pic(src,is_cuda=False)
    cv.imshow('1',reve)
    cv.waitKey(0)