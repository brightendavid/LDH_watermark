#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
hide net 实验有无sobel算子。
is_sobel表示有无加入sobel算子。
直接修改is_sobel 参数即可，其他参数不变
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
        # 使用ycbcr的光照图进行结构检测，未采用（只需要）
        _, x = rgb_to_ycbcr(x)  # 获取y通道作为边缘提取的输入
        x = F.conv2d(x.unsqueeze(1), self.weight, padding=1)

        # x = x[:, 1]
        # x = F.conv2d(x.unsqueeze(1), self.weight, padding=1)
        x = self.norm(x)  # 认为偏色问题在于结构图的干扰

        return x


class HideNet(nn.Module):
    def __init__(self, input_nc, output_nc, nhf=4, norm_layer=None, output_function=nn.Tanh,is_sobel = True):
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
        self.is_sobel = is_sobel
        self.Sobel = Sobel_conv()
        if self.is_sobel:
            self.conv1 = nn.Conv2d(input_nc, nhf, 3, 1, 1)
        else:
            self.conv1 = nn.Conv2d(1, nhf, 3, 1, 1)
        self.conv2 = nn.Conv2d(nhf, output_nc, 1, 1, 0)
        self.conv3 = nn.Conv2d(output_nc + input_nc + 1, output_nc, 3, 1, 1)
        self.output = output_function()
        self.relu = nn.ReLU(True)

        self.norm_layer = norm_layer
        self.norm1 = norm_layer(nhf)
        self.norm2 = norm_layer(output_nc)

    def forward(self, cover, input):
        """

        @param cover: Cover载体图像  有归一化操作  3
        @param input: S的特征  1
        @return:
        """
        # 有cover结构图(3)，原图cover(3)，input_Secret(3)

        if self.is_sobel:
            edge = self.Sobel(cover)
            x = torch.cat((edge, input), 1)  # 堆叠结构图和输入
            x = self.relu(self.norm1(self.conv1(x)))  # in 2; out nhf
            x = self.relu(self.norm2(self.conv2(x)))  # in nhf ;out input
            x2 = torch.cat((x, cover), 1)  # input + output
            x = self.output(self.conv3(x2))  # in 2*input
        else:
            x = input
            x = self.relu(self.norm1(self.conv1(x)))  # in 2; out nhf
            x = self.relu(self.norm2(self.conv2(x)))  # in nhf ;out input
            x2 = torch.cat((x, cover), 1)  # input + output
            x = self.output(self.conv3(x2))  # in 2*input


        return x


if __name__ == '__main__':
    # input_nc = opt.channel_cover * opt.num_cover, output_nc = opt.channel_secret * opt.num_secret, nhf = 64,
    # norm_layer = norm_layer, output_function = nn.Sigmoid
    model = HideNet(input_nc=2,
                    output_nc=1, norm_layer=nn.BatchNorm2d,
                    output_function=nn.Tanh,is_sobel=False).cuda()
    print(model)
    cover = torch.ones((1, 3, 64, 64)).cuda()
    input = torch.ones((1, 1, 64, 64)).cuda()
    b = model(cover, input)
    print("out shape", b.shape)
    # print(b)
    total_params, total_trainable_params = cnn_paras_count(model)
    print(total_params*4/1024,"kb")