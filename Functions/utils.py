#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import random

import torch.nn as nn

import yaml

from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms.functional import *
import cv2 as cv
from JPEG import DiffJPEG

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Averagvalue(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def affine_rotating(img_torch, ang=360):
    """
    ang:角度为-180-180之间
    """
    # 旋转
    angle = (ang * math.pi / 180)
    theta = torch.tensor([
        [math.cos(angle), math.sin(-angle), 0, 0],
        [math.sin(angle), math.cos(angle), 0, 0],
        [0, 0, 1, 0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size()).cuda()
    output = F.grid_sample(img_torch.unsqueeze(0), grid).cuda()
    new_img_torch = output[0].cuda()
    return new_img_torch


def affine_big22(img_torch):
    # 放大 5维  3*4   4维3*3    3维 2*3  1 means this aix is not changed
    theta = torch.tensor([
        [0.5, 0, 0, 0],
        [0, 0.5, 0, 0],
        [0, 0, 1, 0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size()).cuda()
    output = F.grid_sample(img_torch.unsqueeze(0), grid)
    new_img_torch = output[0]
    return new_img_torch


def affine_big23(img_torch):
    # 放大 5维  3*4   4维3*3    3维 2*3  1 means this aix is not changed
    theta = torch.tensor([
        [0.5, 0, 0, 0],
        [0, 0.3, 0, 0],
        [0, 0, 1, 0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size()).cuda()
    output = F.grid_sample(img_torch.unsqueeze(0), grid)
    new_img_torch = output[0]
    return new_img_torch


def affine_big50(img_torch):
    # 放大 5维  3*4   4维3*3    3维 2*3  1 means this aix is not changed
    theta = torch.tensor([
        [0.2, 0, 0, 0],
        [0, 0.1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size()).cuda()
    output = F.grid_sample(img_torch.unsqueeze(0), grid)
    new_img_torch = output[0]
    return new_img_torch


def affine_big_epoch(image_torch, epoch):
    """
    随着epoch增加，放大倍数增加  1-5倍   epoch =0 ,1   ;epoch = 100 =5
    @param image_torch:
    @param epoch:
    @return:
    """
    a = 1 + epoch / 25
    a = np.min([a, 5])  # 最大5倍
    theta = torch.tensor([
        [1 / a, 0, 0, 0],
        [0, 1 / a, 0, 0],
        [0, 0, 1, 0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), image_torch.unsqueeze(0).size()).cuda()
    output = F.grid_sample(image_torch.unsqueeze(0), grid)
    new_image_torch = output[0]
    return new_image_torch


def affine_big05(img_torch):
    # 放大 5维  3*4   4维3*3    3维 2*3  1 means this aix is not changed
    theta = torch.tensor([
        [0.1, 0, 0, 0],
        [0, 0.2, 0, 0],
        [0, 0, 1, 0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size()).cuda()
    output = F.grid_sample(img_torch.unsqueeze(0), grid)
    new_img_torch = output[0]
    return new_img_torch


def affine_big00(img_torch):
    # 放大 5维  3*4   4维3*3    3维 2*3  1 means this aix is not changed
    theta = torch.tensor([
        [0.1, 0, 0, 0],
        [0, 0.1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size()).cuda()
    output = F.grid_sample(img_torch.unsqueeze(0), grid)
    new_img_torch = output[0]
    return new_img_torch


def affine_big55(img_torch):
    # 放大 5维  3*4   4维3*3    3维 2*3  1 means this aix is not changed
    theta = torch.tensor([
        [0.2, 0, 0, 0],
        [0, 0.2, 0, 0],
        [0, 0, 1, 0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size()).cuda()
    output = F.grid_sample(img_torch.unsqueeze(0), grid)
    new_img_torch = output[0]
    return new_img_torch


def affine_small22(img_torch):
    # 缩小  5维  3*4   4维3*3    3维 2*3  1 means this aix is not changed   ,尺度可以变化
    theta = torch.tensor([
        [2, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 1, 0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size()).cuda()
    output = F.grid_sample(img_torch.unsqueeze(0), grid)
    new_img_torch = output[0]
    return new_img_torch


def affine_small23(img_torch):
    # 缩小  5维  3*4   4维3*3    3维 2*3  1 means this aix is not changed   ,尺度可以变化
    theta = torch.tensor([
        [2, 0, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 1, 0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size()).cuda()
    output = F.grid_sample(img_torch.unsqueeze(0), grid)
    new_img_torch = output[0]
    return new_img_torch


def affine_small32(img_torch):
    # 缩小  5维  3*4   4维3*3    3维 2*3  1 means this aix is not changed   ,尺度可以变化
    theta = torch.tensor([
        [3, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 1, 0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size()).cuda()
    output = F.grid_sample(img_torch.unsqueeze(0), grid)
    new_img_torch = output[0]
    return new_img_torch


def change(img):
    # 改变对比度，亮度，色度,只能对image使用，返回Image
    hue_factor = 0.2
    bright_factor = 0.9
    con_factor = 0.7
    img = adjust_brightness(img, bright_factor)
    img = adjust_contrast(img, con_factor)
    img = adjust_hue(img, hue_factor)
    return img


# numpy.random.uniform(low,high,size)
def rnd_brightness(bri, hue):
    # using numpy in cpu
    rnd_hueness = np.random.uniform(-hue, hue, [3, 1, 1])
    rnd_brightness = np.random.uniform(-bri, bri, [1, 1, 1])
    return rnd_hueness + rnd_brightness


def get_rnd_brightness(rnd_bri, rnd_hue, batch_size):
    # 使用正态分布，模拟亮度变化，亮度和对比度公式为：y= ax+b  其中，b表示亮度，a表示对比度，此处只改变亮度
    rnd_hueness = torch.distributions.uniform.Uniform(-rnd_hue, rnd_hue).sample([batch_size, 3, 1, 1])
    rnd_brightness = torch.distributions.uniform.Uniform(-rnd_bri, rnd_bri).sample([batch_size, 1, 1, 1])
    return rnd_hueness + rnd_brightness


def affine_transform(im):
    # 单通道
    from scipy import ndimage
    H = np.array([[1.4, 0.05, -100], [0.05, 1.5, -100], [0, 0, 1]])
    im2 = ndimage.affine_transform(im, H[:2, :2], (H[0, 2], H[1, 2]))
    return im2


def warped(img_torch, t1, t2):
    """
    用于针对pytorch  4维张量的图像扭曲
    @param img_torch: torch.Size([1, 1, 256, 384])  torch.Size([1, 3, 256, 384])
    @return: torch.Size([1, 1, 256, 384])  torch.Size([1, 3, 256, 384])
    """
    # t1 = random.randint(1, 5) / 10  # t从0.1-0.5都没有问题，可能会模糊
    # t2 = random.randint(1, 5) / 10
    theta = torch.tensor([
        [1, t1, 0, 0],
        [t2, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size()).cuda()
    output = F.grid_sample(img_torch.unsqueeze(0), grid)
    new_img_torch = output[0]
    return new_img_torch


# From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    """
        blur_layer = get_gaussian_kernel().cuda()
        blured_img = blur_layer(img_torch)
    @param kernel_size:
    @param sigma:
    @param channels:
    @return:
    """
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def transform_net(encoded_image, args, global_step, batch_size=1):
    """
    noise layer.stegastamp 使用的版本。主要是使用了渐进的扰动强度，代码可读性不是很强.
    没有完全看懂，基本就是设定强度，把强度设计到扰动里面去，没有概率判定.
    扰动强度为关键，原文要求从0开始
    注意，这个noise layer和warp是分离的。warp需要另外加入，已加入
    (warp)->blur_layer->noise->contrast & brightness->saturation->DiffJPEG
    @param encoded_image: contain
    @param args: 设定的超参数，包括loss 的权重，扰动强度,这个东西和命令行还不一样，不能通过'args.+参数'读取参数，
                只能通过args['参数']读取;抄人代码非常麻烦,虽然几乎和实际上的args独立，但是batch之类的要相等
    @param global_step: 现在的epoch
    @return:
    """
    sh = encoded_image.size()
    ramp_fn = lambda ramp: np.min([global_step / ramp, 1.])

    rnd_bri = ramp_fn(args['rnd_bri_ramp']) * args['rnd_bri']
    rnd_hue = ramp_fn(args['rnd_hue_ramp']) * args['rnd_hue']
    # print(ramp_fn(args['rnd_bri_ramp']) )
    rnd_brightness = get_rnd_brightness(rnd_bri, rnd_hue, batch_size)  # [batch_size, 3, 1, 1]:光照,对比度从0开始， 最大值 0.3  0.1
    jpeg_quality = 100. - torch.rand(1)[0] * ramp_fn(args['jpeg_quality_ramp']) * (100. - args['jpeg_quality'])
    # print(jpeg_quality)
    rnd_noise = torch.rand(1)[0] * ramp_fn(args['rnd_noise_ramp']) * args['rnd_noise']

    contrast_low = 1. - (1. - args['contrast_low']) * ramp_fn(args['contrast_ramp'])
    contrast_high = 1. + (args['contrast_high'] - 1.) * ramp_fn(args['contrast_ramp'])
    contrast_params = [contrast_low, contrast_high]

    rnd_sat = torch.rand(1)[0] * ramp_fn(args['rnd_sat_ramp']) * args['rnd_sat']

    # blur(花里胡哨的random blur删除了，换为固定模糊核心)
    blur_layer = get_gaussian_kernel().cuda()
    encoded_image = blur_layer(encoded_image)
    # noise 生成平均值为0，方差随轮数递增的noise,加一个小数防止方差为0，数学上是正确的，就是0矩阵，但是实现中会报错
    noise = torch.normal(mean=0, std=rnd_noise + torch.tensor(0.001).cuda(), size=encoded_image.size(),
                         dtype=torch.float32)
    if args['cuda']:
        noise = noise.cuda()
    encoded_image = encoded_image + noise
    encoded_image = torch.clamp(encoded_image, 0, 1)  # 限制范围到0-1，防止越界

    # contrast & brightness
    contrast_scale = torch.Tensor(encoded_image.size()[0]).uniform_(contrast_params[0], contrast_params[1])
    # 对比度控制，在contrast_low-contrast_high之间的正态分布
    contrast_scale = contrast_scale.reshape(encoded_image.size()[0], 1, 1, 1)
    if args['cuda']:
        contrast_scale = contrast_scale.cuda()
        rnd_brightness = rnd_brightness.cuda()
    encoded_image = encoded_image * contrast_scale
    encoded_image = encoded_image + rnd_brightness
    encoded_image = torch.clamp(encoded_image, 0, 1)

    # saturation
    sat_weight = torch.FloatTensor([.3, .6, .1]).reshape(1, 3, 1, 1)
    if args['cuda']:
        sat_weight = sat_weight.cuda()
    encoded_image_lum = torch.mean(encoded_image * sat_weight, dim=1).unsqueeze_(1)
    encoded_image = (1 - rnd_sat) * encoded_image + rnd_sat * encoded_image_lum

    # jpeg
    # encoded_image = encoded_image.reshape([-1, 3, 400, 400])
    if args['no_jpeg']:
        a, b = encoded_image.shape[2], encoded_image.shape[3]
        J = DiffJPEG.DiffJPEG(width=b, differentiable=True, quality=jpeg_quality - 1, height=a)  # 压缩质量判定,
        # 小心此处的jpeg压缩数值不能是100，否则会出现NaN的情况.出现NaN，网络训练就完蛋了。写这个包的家伙没有考虑到异常处理
        encoded_image = J(encoded_image)

    # 直接加上
    # t1 = random.randint(1, 5) / 10  # t从0.1-0.5都没有问题，可能会模糊
    # t2 = random.randint(1, 5) / 10
    # encoded_image = warped(encoded_image, t1, t2)

    return encoded_image


def get_secret_acc(secret_true, secret_pred):
    if 'cuda' in str(secret_pred.device):
        secret_pred = secret_pred.cpu()
        secret_true = secret_true.cpu()

    secret_pred = torch.round(secret_pred)  # 舍入函数，到整数
    correct_pred = torch.sum((secret_pred - secret_true) == 0, dim=1)
    # print(correct_pred) # true mask
    print(secret_pred.numel())
    print(torch.sum(correct_pred).numpy())
    # str_acc = 1.0 - torch.sum((correct_pred - secret_pred.size()[1]) != 0).numpy() / correct_pred.size()[0] #
    # 不知道在算什么东西
    bit_acc = torch.sum(correct_pred).numpy() / secret_pred.numel()  # .numel() return the num
    return bit_acc


def rgb_to_ycbcr(image: torch.Tensor):
    """
    rgb 转为ycbcr类型，主要是要光照图
    @param image: rgb(tensor)
    @return: ycbcr,y(tensor)
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("input not tensor")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("input size must be (*,3,H,W)")
    r = image[:, 0, :, :]
    g = image[:, 1, :, :]
    b = image[:, 2, :, :]

    delta = 0.5
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564 + delta
    cr = (r - y) * 0.713 + delta
    return torch.stack([y, cb, cr], -3), y


def ycbcr_to_rgb(image: torch.Tensor):
    if not isinstance(image, torch.Tensor):
        raise TypeError("input is not tensor")
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("input size must be (* ,3,H,W)")
    y = image[:, 0, :, :]
    cb = image[:, 1, :, :]
    cr = image[:, 2, :, :]
    delta = 0.5
    cb_shift = cb - delta
    cr_shift = cr - delta

    r = y + 1.403 * cr_shift
    g = y - 0.714 * cr_shift - 0.344 * cb_shift
    b = y + 1.773 * cb_shift
    return torch.stack([r, g, b], -3)


def my_acc_score(prediction, label):
    from sklearn.metrics import accuracy_score
    y = prediction.reshape(-1)
    l = label.reshape(-1)
    y = np.array(y.cpu().detach())
    y = np.where(y > 0.5, 1, 0).astype('int')
    l = np.array(l.cpu().detach()).astype('int')
    return accuracy_score(y_pred=y, y_true=l)


def my_f1_score(prediction, label):  # 输入为两个tensor变量(1, 1, 320, 320) 4维
    from sklearn.metrics import f1_score
    y = prediction.reshape(-1)
    # 转化为向量形式
    l = label.reshape(-1)
    y = np.array(y.cpu().detach())
    y = np.where(y > 0.5, 1, 0).astype('int')
    l = np.array(l.cpu().detach()).astype('int')
    return f1_score(y_pred=y, y_true=l, zero_division=1)


def get_rand_transform_matrix(image_size, d, batch_size):
    Ms = np.zeros((batch_size, 2, 3, 3))
    for i in range(batch_size):
        tl_x = random.uniform(-d, d)  # Top left corner, top
        tl_y = random.uniform(-d, d)  # Top left corner, left
        bl_x = random.uniform(-d, d)  # Bot left corner, bot
        bl_y = random.uniform(-d, d)  # Bot left corner, left
        tr_x = random.uniform(-d, d)  # Top right corner, top
        tr_y = random.uniform(-d, d)  # Top right corner, right
        br_x = random.uniform(-d, d)  # Bot right corner, bot
        br_y = random.uniform(-d, d)  # Bot right corner, right

        rect = np.array([
            [tl_x, tl_y],
            [tr_x + image_size, tr_y],
            [br_x + image_size, br_y + image_size],
            [bl_x, bl_y + image_size]], dtype="float32")

        dst = np.array([
            [0, 0],
            [image_size, 0],
            [image_size, image_size],
            [0, image_size]], dtype="float32")

        M = cv.getPerspectiveTransform(rect, dst)
        M_inv = np.linalg.inv(M)
        Ms[i, 0, :, :] = M_inv
        Ms[i, 1, :, :] = M
    Ms = torch.from_numpy(Ms).float()

    return Ms


def warp_genel(image_input, Ms):
    import torchgeometry
    input_warped = torchgeometry.warp_perspective(image_input, Ms[:, 1, :, :], dsize=(256, 256), flags='bilinear')
    return input_warped


if __name__ == '__main__':
    with open('../cfg/setting.yaml', 'r') as f:
        Hp = yaml.load(f, Loader=yaml.SafeLoader)
    img_path = r'../data/DIV2K_train_HR/0801.png'
    # # img_path = r"F:\DATA\public_dataset\casia\gt\Tp_D_CND_M_N_ani00018_sec00096_00138_gt.png"
    img = cv.imread(img_path)
    img = img[::3,::3]
    img_torch = transforms.ToTensor()(img)
    contain = img_torch.unsqueeze(0).cuda()

    # t1 = random.randint(1, 5) / 10  # t从0.1-0.5都没有问题，可能会模糊
    # t2 = random.randint(1, 5) / 10
    # contain = warped(contain, t1, t2)

    # std_noise = (torch.rand(1) * 0.05).item()
    # noise = torch.randn_like(contain) * std_noise
    # contain = contain + noise
    # brighten_layer = get_rnd_brightness(0.2, 0.2, 1).cuda()
    # contain = contain + brighten_layer
    # blur_layer = get_gaussian_kernel().cuda()
    # contain = blur_layer(contain)
    # a, b = contain.shape[2], contain.shape[3]

    # q = random.randint(0, 10) % 4
    # quality_list = [85, 80, 90, 95]  # 压缩质量不能太低
    # quality = quality_list[q]
    # J = DiffJPEG.DiffJPEG(width=b, differentiable=True, quality=quality, height=a)  # 压缩质量判定
    # contain = J(contain)

    for i in range(0, 100, 5):
        t1 = random.randint(1, 5) / 10  # t从0.1-0.5都没有问题，可能会模糊
        t2 = random.randint(1, 5) / 10
        print(contain.shape)
        img = warped(contain, t1, t2)
        img = img.squeeze(0)
        img = img.cpu().numpy().transpose(1, 2, 0)
        cv.imshow("1", img)
        cv.waitKey(0)

    # img_torch = torch.cat((img_torch,img_torch),0)
    # print(img_torch.shape)
    # # img_torch = torch.rand(1, 3, 1024, 2048).cuda()
    # a,b = img_torch.shape[2], img_torch.shape[3]
    # J = DiffJPEG.DiffJPEG(width=b, differentiable=True, quality=80, height=a)  # 压缩质量判定
    # contain_out = J(img_torch)

    # blur_layer = get_gaussian_kernel().cuda()
    # blured_img = blur_layer(img_torch)
    # print(blured_img.shape)
    # img = blured_img
    # print()

    img = contain.squeeze(0)
    img = img.detach()
    img = img.cpu().numpy().transpose(1, 2, 0)
    # cv.imwrite("gauss.png", img * 255)
    cv.imshow("1", img)
    cv.waitKey(0)
    cv.imwrite("shoot.png", img * 255)
    # name = r"../pics/" + "jpg" + ".jpg"
    # cv.imwrite(name, img * 255)
