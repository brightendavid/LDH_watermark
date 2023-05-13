import math
import os

import numpy as np
from PIL import Image
from scipy.signal import convolve2d

import pandas as pd
import cv2 as cv
from skimage.color import rgb2ycbcr

save_excel_dir = r'C:\Users\brighten\Desktop\rat.xlsx'
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def wrong_message_rat(img, gt):
    img = np.where(img < 125, 0, 1)
    gt =np.where(gt < 125, 0, 1)
    sum = np.sum(np.ones_like(img))
    wrong = np.where(img != gt, 1, 0)
    wrong_sum = np.sum(wrong)
    rat = wrong_sum / sum
    return rat


def wrong_rat_all(dir):
    """
    input:dir
    output:rat_list(list)
    """
    wrong_rat_list = []
    for num in range(0, 90):
        img_path = os.path.join(dir, str(num))
        img_path = img_path + "out_secret.bmp"

        gt_path = os.path.join(dir, str(num))
        gt_path = gt_path + "secrets_in.bmp"

        img = cv.imread(img_path)
        gt = cv.imread(gt_path)

        rat = wrong_message_rat(img, gt)
        wrong_rat_list.append(rat)
    return wrong_rat_list

from skimage.measure import compare_ssim
def SSIM(target, ref, K1=0.01, K2=0.03, gaussian_kernel_sigma=1.5, gaussian_kernel_width=11, L=255):
    # 高斯核，方差为1.5，滑窗为11*11
    gaussian_kernel = np.zeros((gaussian_kernel_width, gaussian_kernel_width))
    for i in range(gaussian_kernel_width):
        for j in range(gaussian_kernel_width):
            gaussian_kernel[i, j] = (1 / (2 * math.pi * (gaussian_kernel_sigma ** 2))) * math.exp(
                -(((i - 5) ** 2) + ((j - 5) ** 2)) / (2 * (gaussian_kernel_sigma ** 2)))

    target = np.array(target, dtype=np.float32)
    ref = np.array(ref, dtype=np.float32)
    ref = np.where(ref < 125, 0, 255)
    if target.shape != ref.shape:
        raise ValueError('输入图像的大小应该一致！')

    target_window = convolve2d(target, np.rot90(gaussian_kernel, 2), mode='valid')
    ref_window = convolve2d(ref, np.rot90(gaussian_kernel, 2), mode='valid')

    mu1_sq = target_window * target_window
    mu2_sq = ref_window * ref_window
    mu1_mu2 = target_window * ref_window

    sigma1_sq = convolve2d(target * target, np.rot90(gaussian_kernel, 2), mode='valid') - mu1_sq
    sigma2_sq = convolve2d(ref * ref, np.rot90(gaussian_kernel, 2), mode='valid') - mu2_sq
    sigma12 = convolve2d(target * ref, np.rot90(gaussian_kernel, 2), mode='valid') - mu1_mu2

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    ssim_array = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim = np.mean(np.mean(ssim_array))

    return ssim


def test_all(dir):
    ssim_list = []
    psnr_list = []
    wrong_rat_list = []
    for num in range(0, 90):
        print(num)
        img_path = os.path.join(dir, str(num))
        img_path = img_path + "out_contain.bmp"

        gt_path = os.path.join(dir, str(num))
        gt_path = gt_path + "out_image.bmp"

        target = Image.open(img_path).convert('L')
        ref = Image.open(gt_path).convert('L')

        psnr = PSNR(target, ref)
        # print('PSNR为:{}'.format(psnr))

        ssim = SSIM(target, ref)
        # print('SSIM为:{}'.format(ssim))

        ssim_list.append(ssim)
        psnr_list.append(psnr)
    wrong_rat_list=wrong_rat_all(dir)
    data = {
        'ssim': ssim_list,
        'psnr': psnr_list,
        'wrogn_rat':wrong_rat_list
    }
    test = pd.DataFrame(data)
    # 按照 acc_sort 降序排序
    # test = test.sort_values(by="acc_sort", ascending=False)
    test.to_excel(save_excel_dir)


def psnrandssim(img1,img2):
    image1 = img1 / 255.0
    image2 = img2 / 255.0
    image1 = rgb2ycbcr(image1)[:, :, 0:1]
    image2 = rgb2ycbcr(image2)[:, :, 0:1]
    image1 = image1 / 255.0
    image2 = image2 / 255.0

    print(image1.shape)
    psnr_val = peak_signal_noise_ratio(image1, image2)
    ssim_val = structural_similarity(image1, image2, win_size=11, gaussian_weights=True, multichannel=True,
                                     data_range=1.0, K1=0.01, K2=0.03, sigma=1.5)
    print("psnr_val", psnr_val)
    print("ssim_val", ssim_val)
    return psnr_val,ssim_val

if __name__ == '__main__':
    target = cv.imread(r'../data/DIV2K_train_HR/0801.png')
    in_gt = cv.imread(r'../data/DIV2K_train_HR/0802.png')
    psnrandssim(target,in_gt)
