#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
基于res net的思路写的
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from Functions.utils import rgb_to_ycbcr
from model_ldh.Revealnet_deep import cnn_paras_count



class Sobel_conv(nn.Module):
    """
    定义sobel算子 卷积,可行
    """

    def __init__(self):
        super(Sobel_conv, self).__init__()
        kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

        self.norm = nn.BatchNorm2d(1)

    def forward(self, x):
        _, x = rgb_to_ycbcr(x)  # 获取y通道作为边缘提取的输入
        x = F.conv2d(x.unsqueeze(1), self.weight, padding=1)
        x = self.norm(x)  # 认为偏色问题在于结构图的干扰
        return x


class HideNet(nn.Module):
    def __init__(self, input_nc, output_nc, nhf=64, norm_layer=None, output_function=nn.Tanh):
        """
        Tanh  -1  1 之间
        @param input_nc:
        @param output_nc:
        @param nhf:
        @param norm_layer:
        @param output_function:
        """
        super(HideNet, self).__init__()
        # input is (3) x 256 x 256
        self.Sobel = Sobel_conv()
        self.norm0 = norm_layer(3)
        self.norm1 = norm_layer(4)
        self.norm2 =norm_layer(output_nc + input_nc - 1)
        self.ConvBlock1 = nn.Sequential(
            nn.Conv2d(input_nc, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(inplace=True)
        )
        self.ConvBlock2 = nn.Sequential(
            nn.Conv2d(nhf, output_nc, 1, 1, 0),
            nn.BatchNorm2d(output_nc),
            nn.ReLU(inplace=True)
        )
        self.res_black1 = nn.Sequential(
            nn.Conv2d(output_nc + input_nc - 1, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(inplace=True)
        )
        self.res_black2 = nn.Sequential(
            nn.Conv2d(nhf, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(inplace=True)
        )
        self.res_black3 = nn.Sequential(
            nn.Conv2d(nhf, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(inplace=True)
        )
        self.res_black4 = nn.Sequential(
            nn.Conv2d(nhf, output_nc, 3, 1, 1),
            nn.BatchNorm2d(output_nc),
            output_function()
        )

    def forward(self,cover,input):
        # 有cover结构图(3)，原图cover(3)，input_Secret(1)
        cover = self.norm0(cover)
        edge = self.Sobel(cover)
        x = torch.cat((edge, input), 1)  # 堆叠结构图和输入
        x = self.norm1(x)

        x = self.ConvBlock1(x)  # in 2; out nhf
        res1 = x
        x = self.ConvBlock2(x)  # in nhf ;out input
        x2 = torch.cat((x, cover), 1)  # input + output
        x2 = self.norm2(x2)
        x = self.res_black1(x2)  # in 2*input
        x = x + res1
        res2 = x
        x = self.res_black2(x)
        x = x + res2
        res3 = x
        x = self.res_black3(x)
        x = x + res3
        x = self.res_black4(x)

        return x


if __name__ == '__main__':
    # input_nc = opt.channel_cover * opt.num_cover, output_nc = opt.channel_secret * opt.num_secret, nhf = 64,
    # norm_layer = norm_layer, output_function = nn.Sigmoid
    model = HideNet(input_nc=4,
                    output_nc=3, norm_layer=nn.BatchNorm2d,
                    output_function=nn.Tanh).cuda()
    print(model)
    cover = torch.ones((1, 3, 64, 64)).cuda()
    input = torch.ones((1, 3, 64, 64)).cuda()
    b = model(cover, input)
    print("out shape", b.shape)
    # print(b)
    total_params, total_trainable_params = cnn_paras_count(model)
    print(total_params * 4 / 1024, "kb")
