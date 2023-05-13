import torch
import torch.nn as nn
import torch.nn.functional as F

from Functions.utils import rgb_to_ycbcr
from model_ldh.Revealnet_deep import cnn_paras_count

"""
    im = Image.open('./data/DIV2K_train_HR/0802.png').convert('L')
    # 将图片数据转换为矩阵
    im = np.array(im, dtype='float32')
    # 将图片矩阵转换为pytorch tensor,并适配卷积输入的要求
    im = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1])))
    # 边缘检测操作
    # edge_detect = nn_conv2d(im)
    edge_detect = functional_conv2d(im)
    # 将array数据转换为image
    im = Image.fromarray(edge_detect)
    # image数据转换为灰度模式
    im = im.convert('L')
    # 保存图片
    im.save('edge.jpg', quality=95)
"""


class Sobel_conv(nn.Module):
    """
    定义sobel算子 卷积,可行
    """

    def __init__(self):
        super(Sobel_conv, self).__init__()
        kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

        # self.norm = nn.BatchNorm2d(3)

    def forward(self, x):
        # 使用ycbcr的光照图进行结构检测，未采用（只需要）
        # _, x = rgb_to_ycbcr(x)  # 获取y通道作为边缘提取的输入

        # x = F.conv2d(x.unsqueeze(1), self.weight, padding=1)

        x = x[:, 1]
        x = F.conv2d(x.unsqueeze(1), self.weight, padding=1)
        # x = self.norm(x)  # 认为偏色问题在于结构图的干扰

        return x


class HideNet(nn.Module):
    def __init__(self, input_nc, output_nc, nhf=4, norm_layer=None, output_function=nn.Tanh):
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
        self.conv1 = nn.Conv2d(input_nc, nhf, 3, 1, 1)
        self.conv2 = nn.Conv2d(nhf, output_nc, 1, 1, 0)
        self.conv3 = nn.Conv2d(output_nc + input_nc + 1, output_nc, 3, 1, 1)
        self.output = output_function()
        self.relu = nn.ReLU(True)

        self.norm_layer = norm_layer
        if norm_layer != None:
            self.norm1 = norm_layer(nhf)
            self.norm2 = norm_layer(output_nc)

    def forward(self, cover, input):
        """

        @param cover: Cover载体图像  有归一化操作  3
        @param input: S的特征  1
        @return:
        """
        # 有cover结构图(3)，原图cover(3)，input_Secret(3)
        edge = self.Sobel(cover)
        x = torch.cat((edge, input), 1)  # 堆叠结构图和输入

        if self.norm_layer != None:
            x = self.relu(self.norm1(self.conv1(x)))  # in 2; out nhf
            x = self.relu(self.norm2(self.conv2(x)))  # in nhf ;out input
            x2 = torch.cat((x, cover), 1)  # input + output
            x = self.output(self.conv3(x2))  # in 2*input
        else:
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))

            x2 = torch.cat((x, cover), 1)
            x = self.output(self.conv3(x2))

        return x


if __name__ == '__main__':
    # input_nc = opt.channel_cover * opt.num_cover, output_nc = opt.channel_secret * opt.num_secret, nhf = 64,
    # norm_layer = norm_layer, output_function = nn.Sigmoid
    model = HideNet(input_nc=4,
                    output_nc=3, norm_layer=nn.BatchNorm2d,
                    output_function=nn.Tanh).cuda()
    print(model)
    cover = torch.ones((1, 3, 64, 64)).cuda()
    input = torch.ones((1, 1, 64, 64)).cuda()
    b = model(cover, input)
    print("out shape", b.shape)
    # print(b)
    total_params, total_trainable_params = cnn_paras_count(model)
    print(total_params*4/1024,"kb")