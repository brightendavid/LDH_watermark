#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
2022/4/10
测试数据集中的图像
"""
import random
import sys
import pandas as pd
from Functions.utils import affine_rotating, get_rnd_brightness, get_gaussian_kernel, \
    warped, my_acc_score, my_f1_score, affine_small22, affine_big22
from JPEG import DiffJPEG
from test.ssim_psnr import psnrandssim

sys.path.append('../')
import cv2 as cv
import torch.utils.data.dataloader
from Dataset.dataloader import Mydataset
from Functions.loss_functions import *
from model_ldh import Revealnet_deep, Hide_net, Encode_net, Revealnet
import numpy as np

# model_path0 = r"F:\watermark_weight\weight\0709 -best\BEST_Encode_07-09watermark_checkpoint30.pth" # 差不多能够用
# model_path0 = r"E:\weight\Encode_07-01watermark_checkpoint83.pth" # 变色
# model_path0 = r"C:\Users\brighten\Desktop\Encode_07-09watermark_checkpoint80.pth"
# model_path0 = r"E:\weight\0707\Encode_07-07watermark_checkpoint65.pth" # NL
# model_path0 = r"E:\weight\0716\BEST_Encode_07-16watermark_add_noise_checkpoint126.pth"
# model_path0 = r"F:\watermark_weight\weight\0710\BEST_Encode_07-10watermark_checkpoint49.pth"  # paper
# model_path0 = r'F:\watermark_weight\0726\BEST_Encode_07-26watermark_TObelast_checkpoint14.pth'
# model_path0 = r"F:\watermark_weight\weight\Encode_07-01watermark_checkpoint83.pth"
# model_path0 = r"F:\watermark_weight\weight\0707\Encode_07-07watermark_checkpoint65.pth"
# model_path0 = r"F:\watermark_weight\weight\0631\Encode_06-29watermark_checkpoint36.pth"
# model_path0 = r"F:\watermark_weight\0806\BEST_Encode_08-06watermark_lunwen_checkpoint104.pth" # 去除sobel
# model_path0 = r"F:\watermark_weight\0801\Encode_08-01watermark_noiselayer shiyan_checkpoint90.pth"
model_path0 = r"F:\watermark_weight\weight\0717_deep_reveal\BEST_Encode_07-17watermark_add_noise_checkpoint24.pth"
save_path = r"C:\Users\brighten\Desktop\compare_OUR_AT.xlsx"


def main():
    usedata = 'des'  # 如dataloader中所示，使用的是 F盘下的val文件夹，确实是test图像；训练使用的是train文件夹
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model_path1 = model_path0.replace("Encode", "Hide")
    model_path2 = model_path0.replace("Encode", "Rev")
    checkpoint0 = torch.load(model_path0, map_location=device)
    checkpoint1 = torch.load(model_path1, map_location=device)
    checkpoint2 = torch.load(model_path2, map_location=device)

    torch.cuda.empty_cache()
    testData = Mydataset(device=usedata, train_val_test_mode='test')
    testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=1, num_workers=0)
    # 0725前
    Enet = Encode_net.UnetGenerator(input_nc=1,
                                    output_nc=1, num_downs=5, norm_layer=nn.BatchNorm2d,
                                    output_function=nn.Sigmoid)
    Hnet = Hide_net.HideNet(input_nc=2,
                            output_nc=3, norm_layer=nn.BatchNorm2d,
                            output_function=nn.Tanh).cuda()  # 有is_sobel参数
    Rnet = Revealnet.RevealNet(input_nc=3,
                               output_nc=1, norm_layer=nn.BatchNorm2d,
                               output_function=nn.Sigmoid)

    # 0725 之后
    # Enet = Encode_net.UnetGenerator(input_nc=1,
    #                                 output_nc=3, num_downs=5, norm_layer=nn.BatchNorm2d,
    #                                 output_function=nn.Sigmoid)
    # Hnet = Hide_net_deep.HideNet(input_nc=4,
    #                              output_nc=3, norm_layer=nn.BatchNorm2d,
    #                              output_function=nn.Tanh).cuda()
    # Rnet = Revealnet.RevealNet(input_nc=3,
    #                            output_nc=1, norm_layer=nn.BatchNorm2d,
    #                            output_function=nn.Sigmoid)

    if torch.cuda.is_available():
        Enet.cuda()
        Hnet.cuda()
        Rnet.cuda()
    else:
        Enet.cpu()
        Hnet.cpu()
        Rnet.cpu()
    try:
        Enet.load_state_dict(checkpoint0['state_dict'])
        Hnet.load_state_dict(checkpoint1['state_dict'])
        Rnet.load_state_dict(checkpoint2['state_dict'])
    except KeyError:
        Enet.load_state_dict(checkpoint0)
        Hnet.load_state_dict(checkpoint1)
        Rnet.load_state_dict(checkpoint2)
    Hnet.eval()
    Enet.eval()
    Rnet.eval()

    # 对于保存了优化器的模型，重新保存，只保存权重state_dict()
    # torch.save(Enet.state_dict(), model_path0[:-4]+"dict"+model_path0[-4:])
    # torch.save(Hnet.state_dict(), model_path1[:-4]+"dict"+model_path0[-4:])
    # torch.save(Rnet.state_dict(), model_path2[:-4]+"dict"+model_path0[-4:])

    test(Enet=Enet, Hnet=Hnet, Rnet=Rnet, dataParser=testDataLoader)  # test直接用val就行


def test(Enet, Hnet, Rnet, dataParser):  # 测试
    is_jpeg = False
    is_bright = False
    is_blur = False
    is_bigsmall = False
    is_warp = False
    is_noise = False
    is_rot = False
    is_psnrssim = False  # 这个比较慢
    is_save_pic = True
    is_acc_f1 = False
    is_rectangle_and_big = False

    Enet.eval()
    Hnet.eval()
    Rnet.eval()

    acc_list = []
    psnr_list = []
    ssim_list = []
    f1_list = []
    acc_list_jpeg = [[], [], [], []]
    f1_list_jpeg = [[], [], [], []]

    for batch_index, input_data in enumerate(dataParser):
        print(batch_index)
        images = input_data['image'].cuda()
        secrets_in = input_data['secret_in'].cuda()

        secret = input_data['secret_out'].cuda()
        cover = input_data['contain'].cuda()
        with torch.set_grad_enabled(False):
            images.requires_grad = False
            encode_S = Enet(secrets_in).cuda()  # 生成的残差 输入的是secrets
            res = Hnet(cover, encode_S)  # 3通道

            # res = torch.where(secrets_in == 1, res, torch.zeros_like(res).cuda())
            contain = res + cover
            # res = affine_big22(res)
            if is_rot:
                ang = 20
                contain = affine_rotating(contain, ang=ang)
                secret = affine_rotating(secret, ang=ang)
            if is_noise:
                std_noise = (torch.rand(1) * 0.05).item()
                noise_layer = torch.randn_like(contain) * std_noise
                contain = contain + noise_layer
            if is_bright:
                # 加入光照和对比度变化
                brighten_layer = get_rnd_brightness(0.1, 0.1, 1).cuda()
                contain = contain + brighten_layer
            if is_bigsmall:
                contain = affine_small22(contain)
            if is_blur:
                # gaussian blur
                blur_layer = get_gaussian_kernel().cuda()
                contain = blur_layer(contain)

            if is_warp:
                # 图像扭曲
                t1 = random.randint(1, 5) / 10  # t从0.1-0.5都没有问题，可能会模糊
                t2 = random.randint(1, 5) / 10
                contain = warped(contain, t1, t2)
                secret = warped(secret, t1, t2)

            if is_jpeg:
                # jpeg压缩实验，可微类型，设置多个quality，画表
                quality_list = [60, 70, 80, 90]
                a, b = contain.shape[2], contain.shape[3]
                for i in range(len(quality_list)):
                    J = DiffJPEG.DiffJPEG(width=b, differentiable=True, quality=quality_list[i], height=a)  # 压缩质量判定
                    contain_jpeg = (J(contain))
                    secret_out = Rnet(contain_jpeg)  # recover

                    acc = my_acc_score(secret_out, secret)
                    f1 = my_f1_score(secret_out, secret)
                    acc_list_jpeg[i].append(acc)
                    f1_list_jpeg[i].append(f1)

            # a, b = contain.shape[2], contain.shape[3]
            # J = DiffJPEG.DiffJPEG(width=b, differentiable=True, quality=80, height=a)  # 压缩质量判定
            # contain = (J(contain))
            secret_out = Rnet(contain)  # recover

            # 生成cpu下的图像
            out_image = tensor2np(images)
            res = tensor2np(res)
            if is_bigsmall:
                out_secret = tensor2np(secret_out) * 20
            else:
                out_secret = tensor2np(secret_out)
            out_contain = tensor2np(contain)
            secrets_in = tensor2np(secrets_in)

            if is_acc_f1:
                acc = my_acc_score(secret_out, secret)
                f1 = my_f1_score(secret_out, secret)
                acc_list.append(acc)
                f1_list.append(f1)
            else:
                acc_list.append('acc')
                f1_list.append('f1')

            if is_psnrssim:
                psnr_val, ssim_val = psnrandssim(out_contain, out_image)
                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)
            else:
                psnr_list.append('psnr')
                ssim_list.append('ssim')

            if is_save_pic:
                if is_rectangle_and_big:
                    # 放大，并框出，显示在左下角
                    out_contain_big = cv.resize(out_contain, None, None, fx=8, fy=8, interpolation=cv.INTER_LINEAR)
                    out_contain[300:511, 0:300] = out_contain_big[500:711, 500:800]

                    out_image_big = cv.resize(out_image, None, None, fx=8, fy=8, interpolation=cv.INTER_LINEAR)
                    out_image[300:511, 0:300] = out_image_big[500:711, 500:800]

                    out_contain = np.ascontiguousarray(out_contain)
                    out_image = np.ascontiguousarray(out_image)
                    pt1 = (500 // 8, 500 // 8)
                    pt2 = (800 // 8, 711 // 8)
                    cv.rectangle(out_image, pt1, pt2, (0, 0, 255), 2)
                    cv.rectangle(out_contain, pt1, pt2, (0, 0, 255), 2)

                    pt3 = (0, 300)
                    pt4 = (300, 511)
                    cv.line(out_image, pt1, pt3, (0, 0, 255), 1)
                    cv.line(out_image, pt4, pt2, (0, 0, 255), 1)

                    cv.line(out_contain, pt1, pt3, (0, 0, 255), 1)
                    cv.line(out_contain, pt4, pt2, (0, 0, 255), 1)

                    cv.rectangle(out_image, pt3, pt4, (0, 0, 255), 2)
                    cv.rectangle(out_contain, pt3, pt4, (0, 0, 255), 2)

                cv.imwrite(r"./save_result/" + str(batch_index) + "res" + ".bmp", res)
                cv.imwrite(r"./save_result/" + str(batch_index) + "secrets_in" + ".bmp", secrets_in)
                cv.imwrite(r"./save_result/" + str(batch_index) + "out_secret" + ".bmp", out_secret)
                cv.imwrite(r"./save_result/" + str(batch_index) + "out_contain" + ".bmp", out_contain)
                cv.imwrite(r"./save_result/" + str(batch_index) + "out_image" + ".bmp", out_image)
    if not is_jpeg:
        data = {
            'acc': acc_list,
            'f1': f1_list,
            'ssim': ssim_list,
            'psnr': psnr_list,
        }
    else:
        data = {
            'acc': acc_list,
            'f1': f1_list,
            'ssim': ssim_list,
            'psnr': psnr_list,
            'jpeg60': f1_list_jpeg[0],
            'jpeg70': f1_list_jpeg[1],
            'jpeg80': f1_list_jpeg[2],
            'jpeg90': f1_list_jpeg[3],
        }

    # test = pd.DataFrame(data)
    # test.to_excel(save_path)


def tensor2np(src):
    output = src.squeeze(0)
    output = np.array(output.cpu().detach().numpy(), dtype='float32')
    output = np.transpose(output, (1, 2, 0))
    output *= 255.0
    output = np.array(output, dtype=np.float32)
    return output


def clean_result():
    # 清空save_result
    import os
    import shutil
    path = r'./save_result'
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)


if __name__ == '__main__':
    # clean_result()
    main()
