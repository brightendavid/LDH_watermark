# encoding: utf-8
# 解码器，从3通道图像中提取 单通道的二值图
# 改为Unet + conv*7的结构
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    双重的   conc+BN+Relu
    (convolution => [BN] => ReLU) * 2
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            # 双重的卷积，所以有三参数
            #  in_channels, out_channels, mid_channels=None  第三个参数默认是空。如果设置bilinear=Ture,那么就
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)  # 反卷积
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])  # 外围加像素点，使得X1,X2的大小相等
        # if you have padding issues, see

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class RevealNet(nn.Module):
    # net 1 和原有的Unet差不多
    # 输出为4个特征  分别为  1阶段条带
    def __init__(self, n_channels=3, bilinear=False):
        super(RevealNet, self).__init__()
        factor = 2 if bilinear else 1
        _factor = 1 if bilinear else 2
        # print('factor is : ',_factor)
        self.n_channels = n_channels
        self.n_classes = 1
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 1)  # stage1 的输出不是二值化的，而是分布在0-1之间的  用的激活函数是sigmoid 函数

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # stage_x1 = self.up1(x5, x4)
        stage_x2 = self.up2(x4, x3)
        stage_x3 = self.up3(x3, x2)
        stage_x4 = self.up4(stage_x3, x1)
        logits = self.outc(stage_x4)
        return logits

class OutConv(nn.Module):
    # 单卷积和sigmoid 激活
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.Sigmoid()(x)
        return x


def cnn_paras_count(net):
    """cnn参数量统计, 使用方式cnn_paras_count(net)"""
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    return total_params, total_trainable_params


if __name__ == '__main__':
    # input_nc = opt.channel_cover * opt.num_cover, output_nc = opt.channel_secret * opt.num_secret, nhf = 64,
    # norm_layer = norm_layer, output_function = nn.Sigmoid
    model = RevealNet(3,bilinear=False)
    a = torch.zeros((1, 3, 356, 200))
    b = model(a)
    print("out shape", b.shape)
    total_params, total_trainable_params = cnn_paras_count(model)
    print(total_params*4/1024/1024,"mb")