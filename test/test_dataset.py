#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
2022/4/10
测试数据集中的图像
"""
import sys

from Functions.utils import affine_rotating, get_rnd_brightness, affine_big22, affine_small32

sys.path.append('../')
import cv2 as cv
import torch.nn as nn
import torch.utils.data.dataloader
from Dataset.dataloader import Mydataset
from Functions.loss_functions import *
from model_udh import Revealnet, Hide
import numpy as np

model_path1 = r'../save_model/0710/BEST_Encode_07-10watermark_checkpoint49dict.pth'
model_path2 = r'../save_model/0710/BEST_Rev_07-10watermark_checkpoint49dict.pth'


def main():
    usedata = 'sjw'
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    checkpoint1 = torch.load(model_path1, map_location=device)
    checkpoint2 = torch.load(model_path2, map_location=device)
    torch.cuda.empty_cache()
    testData = Mydataset(device=usedata, train_val_test_mode='train')
    testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=1, num_workers=0)
    Hnet = Hide.UnetGenerator(input_nc=1,
                              output_nc=3 * 1, num_downs=5, norm_layer=nn.BatchNorm2d,
                              output_function=nn.Sigmoid)
    Rnet = Revealnet.RevealNet(input_nc=3,
                               output_nc=1, norm_layer=nn.BatchNorm2d,
                               output_function=nn.Sigmoid)
    if torch.cuda.is_available():
        Hnet.cuda()
        Rnet.cuda()
    else:
        Hnet.cpu()
        Rnet.cpu()

    Hnet.load_state_dict(checkpoint1['state_dict'])
    Rnet.load_state_dict(checkpoint2['state_dict'])
    Hnet.eval()
    Rnet.eval()

    test(Hnet=Hnet, Rnet=Rnet, dataParser=testDataLoader)  # test直接用val就行


def test(Hnet, Rnet, dataParser):  # 测试
    Hnet.eval()
    Rnet.eval()
    for batch_index, input_data in enumerate(dataParser):
        images = input_data['image'].cuda()
        secrets_in = input_data['secret_in'].cuda()
        with torch.set_grad_enabled(False):
            images.requires_grad = False
            res = Hnet(secrets_in)
            container = images + res

            # ang = 140
            # container = affine_rotating(container, ang=ang)

            # std_noise = (torch.rand(1) * 0.05).item()
            # noise_layer = torch.randn_like(container) * std_noise
            # container = container + noise_layer
            # # 加入光照和对比度变化
            # brighten_layer = get_rnd_brightness(0.4, 0.3, 1).cuda()
            # container = container + brighten_layer

            # container = affine_big22(container)

            out_sec = Rnet(container)
            out_image = tensor2np(images)
            res = abs(container - images)
            res=tensor2np(res)
            out_secret = tensor2np(out_sec)
            out_contain = tensor2np(container)
            secrets_in= tensor2np(secrets_in)
            cv.imwrite(r"./save_result/" + str(batch_index) + "res" + ".bmp", res)
            cv.imwrite(r"./save_result/" + str(batch_index) + "secrets_in" + ".bmp", secrets_in)
            cv.imwrite(r"./save_result/" + str(batch_index) + "out_secret" + ".bmp", out_secret)
            cv.imwrite(r"./save_result/" + str(batch_index) + "out_contain" + ".bmp", out_contain)
            cv.imwrite(r"./save_result/" + str(batch_index) + "out_image" + ".bmp", out_image)


def main_DDH():
    usedata = 'sjw'
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    checkpoint1 = torch.load(model_path1, map_location=device)
    checkpoint2 = torch.load(model_path2, map_location=device)
    torch.cuda.empty_cache()
    testData = Mydataset(device=usedata, train_val_test_mode='train')
    testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=1, num_workers=0)
    Hnet = Hide.UnetGenerator(input_nc=4,
                              output_nc=3 * 1, num_downs=5, norm_layer=nn.BatchNorm2d,
                              output_function=nn.Sigmoid)
    Rnet = Revealnet.RevealNet(input_nc=3,
                               output_nc=1, norm_layer=nn.BatchNorm2d,
                               output_function=nn.Sigmoid)
    if torch.cuda.is_available():
        Hnet.cuda()
        Rnet.cuda()
    else:
        Hnet.cpu()
        Rnet.cpu()

    Hnet.load_state_dict(checkpoint1['state_dict'])
    Rnet.load_state_dict(checkpoint2['state_dict'])
    Hnet.eval()
    Rnet.eval()

    test_DDH(Hnet=Hnet, Rnet=Rnet, dataParser=testDataLoader)  # test直接用val就行


def test_DDH(Hnet, Rnet, dataParser):  # 测试
    Hnet.eval()
    Rnet.eval()
    for batch_index, input_data in enumerate(dataParser):
        images = input_data['image'].cuda()
        secrets_in = input_data['secret_in'].cuda()
        with torch.set_grad_enabled(False):
            images.requires_grad = False
            inputs = torch.cat((secrets_in, images), 1)  # torch.Size([1, 4, 256, 256])
            Hide_out = Hnet(inputs).cuda()  # 生成的残差 输入的是secrets
            container = Hide_out.cuda()  # 获取含有S的Cover

            # ang = 140
            # container = affine_rotating(container, ang=ang)

            # std_noise = (torch.rand(1) * 0.05).item()
            # noise_layer = torch.randn_like(container) * std_noise
            # container = container + noise_layer
            # # 加入光照和对比度变化
            brighten_layer = get_rnd_brightness(0.5, 0.3, 1).cuda()
            container = container + brighten_layer
            # container = affine_big22(container)

            # contain = container[:,:,0:200,0:200]
            # out_sec = Rnet(contain)
            out_sec = Rnet(container)
            res = abs(container - images)

            out_image = tensor2np(images)
            res = tensor2np(res)
            secrets_in = tensor2np(secrets_in)
            out_secret = tensor2np(out_sec)
            out_contain = tensor2np(container)
            cv.imwrite(r"./save_result/" + str(batch_index) + "res" + ".bmp", res)
            cv.imwrite(r"./save_result/" + str(batch_index) + "secrets_in" + ".bmp", secrets_in)
            cv.imwrite(r"./save_result/" + str(batch_index) + "out_secret" + ".bmp", out_secret)
            cv.imwrite(r"./save_result/" + str(batch_index) + "out_contain" + ".bmp", out_contain)
            cv.imwrite(r"./save_result/" + str(batch_index) + "out_image" + ".bmp", out_image)


def tensor2np(src):
    output = src.squeeze(0)
    output = np.array(output.cpu().detach().numpy(), dtype='float32')
    output = np.transpose(output, (1, 2, 0))
    output *= 255.0
    return output


if __name__ == '__main__':
    main()
