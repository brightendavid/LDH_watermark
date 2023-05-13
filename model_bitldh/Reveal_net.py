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


class StegaStampDecoder(nn.Module):
    def __init__(self, msg_size=100, height=400, width=400):
        super(StegaStampDecoder, self).__init__()

        # 空间变换器定位 - 网络
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True),
        )
        # 3 * 2 affine矩阵的回归量
        self.fc_loc = nn.Sequential(
            nn.Linear(320000, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 2))

        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(21632, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, msg_size),
            nn.Sigmoid()
        )

        self.initialize_weights()

    def initialize_weights(self):
        # 定义权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                m.bias.data.zero_()

        # 使用身份转换初始化权重/偏差
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))

    def forward(self, x, use_stn=False):
        x = x - .5

        # 空间变换器网络转发功能
        # 需要待 decoder 部分稳定才可以开启 STN(Spatial transformer networks)空间转换网络
        if use_stn:
            xs = self.localization(x)
            xs = xs.reshape(-1, 320000)
            theta = self.fc_loc(xs)
            theta = theta.reshape(-1, 2, 3)  # （?,2,3）的仿射矩阵

            grid = F.affine_grid(theta, x.size(), align_corners=True)  # 对tensor 进行仿射变换，进行矫正
            x = F.grid_sample(x, grid, align_corners=True)  # 双线性插值

        return self.decoder(x), x + .5


if __name__ == '__main__':
    # input_nc = opt.channel_cover * opt.num_cover, output_nc = opt.channel_secret * opt.num_secret, nhf = 64,
    # norm_layer = norm_layer, output_function = nn.Sigmoid
    pass
