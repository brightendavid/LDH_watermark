# Pytorch
import cv2 as cv
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import *

# Local
import sys

from JPEG.decompression import decompress_jpeg
from JPEG.compression import compress_jpeg
from JPEG.utils import diff_round, quality_to_factor

"""
could only be used for a single 3 channels picture?
"""


class DiffJPEG(nn.Module):
    def __init__(self, height, width, differentiable=True, quality=80):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = compress_jpeg(rounding=rounding, factor=factor).cuda()
        self.decompress = decompress_jpeg(height, width, rounding=rounding,
                                          factor=factor).cuda()

    def forward(self, x):
        '''

        '''
        y, cb, cr = self.compress(x)  # compress 没有问题
        recovered = self.decompress(y, cb, cr)
        return recovered


def testt():
    img_path = r'../data/DIV2K_train_HR/0801.png'
    img = cv.imread(img_path)
    img = img[0:512, 0:512]
    a, b = img.shape[0], img.shape[1]
    img_torch = transforms.ToTensor()(img)
    print(img_torch.shape)
    img_torch = img_torch.unsqueeze(0)
    for i in [10, 20, 30, 50, 60, 80, 99]:
        J = DiffJPEG(width=b, differentiable=True, quality=i, height=a)
        BB = J(img_torch)
        img = BB.squeeze(0)
        img = (img.detach().numpy().transpose(1, 2, 0))
        cv.imwrite(str(i) + ".png", img * 255)


if __name__ == '__main__':
    img_torch = torch.randn(1, 3, 192, 160).cuda()
    a, b = img_torch.shape[2], img_torch.shape[3]
    J = DiffJPEG(width=b, differentiable=True, quality=40, height=a)
    BB = J(img_torch)
    print(BB.shape)
