import torch.nn as nn
import torch
import torch.nn.functional as F
from math import exp


def cross_entropy_loss(prediction, label):
    # 交叉熵 期望和预期的差值
    label = label.long()
    mask = (label != 0).float()  # 带有权重的交叉熵 计算前景和背景的像素个数

    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    mask[mask != 0] = num_negative / (num_positive + num_negative)  # 0.995
    mask[mask == 0] = num_positive / (num_positive + num_negative)  # 0.005
    cost = torch.nn.functional.binary_cross_entropy(
        prediction.float(), label.float(), weight=mask)
    return cost


def entropy_loss(pred):
    # 信息熵loss 无监督loss
    # -(x*log(x)+(1-x)*log(1-x))
    pred = torch.clamp(pred, 1e-4, 1 - 1e-4)  # 不能太接近0 1 会越界
    one = torch.ones_like(pred).cuda()
    one_jian_x = one - pred
    loss = -(torch.mean(pred * torch.log(pred)) + torch.mean(one_jian_x * torch.log(one_jian_x)))  # 防止越界
    return loss


# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


def cosine_distance(p, sp, dim=1, eps=1e-5):
    """
    这个写的有些问题
    @param p:
    @param sp:
    @param dim:
    @param eps:
    @return:
    """
    cosine_sim = nn.CosineSimilarity(dim=dim, eps=eps)
    d, c, a, b = p.shape
    tensor_shape = (a * b * c)
    return torch.sum((torch.tensor(1).cuda() - cosine_sim(p, sp)) / torch.tensor(2).cuda()) / (a * b * c) / tensor_shape
    # 接近0 图像颜色角越接近,越小越好   (1-（-1，1）)/2 = [0,1]


class color_Loss(nn.Module):
    def __init__(self):
        super(color_Loss, self).__init__()

    def forward(self, x, y):
        b, c, h, w = x.shape

        x = x.view(b, c, h * w)
        y = y.view(b, c, h * w)
        up = x * y
        up = torch.sum(up, dim=1)
        x_2 = x ** 2
        y_2 = y ** 2
        down1 = (x_2[:, 0, :] + x_2[:, 1, :] + x_2[:, 2, :] + 10e-6) ** 0.5
        down2 = (y_2[:, 0, :] + y_2[:, 1, :] + y_2[:, 2, :] + 10e-6) ** 0.5
        down = down1 * down2
        down = down + 10e-6
        out = up / down
        return 1-out.mean()


def rgb_diff_loss(res):
    """
    res in [0,1]
    要求输出的残差图  的rgb三个通道接近
    实际效果没有用
    @param res: res为hide net 输出的残差(*,3,W,H)
    @return: loss  in range[0,1]    max is (0,0.5,1); min is (x,x,x)
    """
    res_size = res.shape[2] * res.shape[3] * 2
    r = res[:, 0, :, :]
    g = res[:, 1, :, :]
    b = res[:, 2, :, :]
    loss = torch.sum(torch.abs(r - g) + torch.abs(r - b) + torch.abs(g - b)) / res_size
    return loss


if __name__ == '__main__':
    a = torch.ones((1, 3, 64, 64)).cuda()
    b = torch.zeros((1, 3, 64, 64)).cuda()
    # c = torch.cat((a, a * 0.5, a * 0), 1)
    # print(rgb_diff_loss(c))
    # color_Loss =color_Loss()
    # print(color_Loss(a,b))
    # print(cosine_distance(a,b,0))