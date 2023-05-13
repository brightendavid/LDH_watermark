#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
2022/4/10
测试数据集中的图像
in fact ,in a work for hide or reveal ,we only need to load one model_udh ,rather than load all modles.
"""
from torchvision import transforms
from Dataset.secret_data import *
sys.path.append('../')
import cv2 as cv
import torch.utils.data.dataloader
from Functions.loss_functions import *
from model_ldh import Revealnet, Hide_net, Encode_net
import numpy as np

model_path0 = r"../save_model/0710/BEST_Encode_07-10watermark_checkpoint49dict.pth"
# model_path0 = r"F:\watermark_weight\0801\Encode_08-01watermark_noiselayer shiyan_checkpoint90.pth"

h = 544
w = 960


def Reveal(data, is_cuda=False):
    model_path2 = model_path0.replace("Encode", "Rev")
    # 大约可以使用2000*2000分辨率的图像进行水印提取
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    checkpoint2 = torch.load(model_path2, map_location=device)
    torch.cuda.empty_cache()
    Rnet = Revealnet.RevealNet(input_nc=3,
                               output_nc=1, norm_layer=nn.BatchNorm2d,
                               output_function=nn.Sigmoid)
    if is_cuda:
        Rnet.cuda()
    else:
        Rnet.cpu()
    try:
        Rnet.load_state_dict(checkpoint2['state_dict'])
    except KeyError:
        Rnet.load_state_dict(checkpoint2)
    Rnet.eval()
    out_sec = Rnet(data)
    out_image = tensor2np(out_sec)
    return out_image


def Hide_secret(data, secret="11111"):
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    torch.cuda.empty_cache()
    Enet = Encode_net.UnetGenerator(input_nc=1,
                                    output_nc=1, num_downs=5, norm_layer=nn.BatchNorm2d,
                                    output_function=nn.Sigmoid).cuda()
    Hnet = Hide_net.HideNet(input_nc=2,
                            output_nc=3, norm_layer=nn.BatchNorm2d,
                            output_function=nn.Tanh).cuda()
    model_path1 = model_path0.replace("Encode", "Hide")

    checkpoint0 = torch.load(model_path0, map_location=device)
    checkpoint1 = torch.load(model_path1, map_location=device)
    Hnet.load_state_dict(checkpoint1)
    Enet.load_state_dict(checkpoint0)
    Enet.eval()
    Hnet.eval()
    secret = gen_data(word=secret)
    print(secret.shape)
    secret = secret[:h, :w]
    data = transforms.Compose([
        transforms.ToTensor()
    ])(data)  # 张量化
    data = data[np.newaxis, :, :, :]
    data = data.cuda()
    secret = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])(secret)  # 张量化
    secret = secret[np.newaxis, :, :, :]
    secret = secret.cuda()

    global start_Enet
    start_Enet = datetime.now()
    encode_S = Enet(secret).cuda()  # 生成的残差 输入的是secrets

    global start_Hnet
    start_Hnet = datetime.now()

    res = Hnet(data, encode_S)  # 3通道
    res1 = tensor2np(res)
    cv.imwrite("test_picture/res.png", res1 * 15)
    out_contain = res + data

    out_image = tensor2np(out_contain)
    return out_image


def Hide_secret_Enet_only(secret="11111"):
    """
    只计算Enet
    @param secret:
    @return:
    """
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    torch.cuda.empty_cache()
    Enet = Encode_net.UnetGenerator(input_nc=1,
                                    output_nc=1, num_downs=5, norm_layer=nn.BatchNorm2d,
                                    output_function=nn.Sigmoid).cuda()
    checkpoint0 = torch.load(model_path0, map_location=device)
    Enet.load_state_dict(checkpoint0)
    Enet.eval()
    print(secret)
    secret = gen_data(word=secret,w =w,h=h)
    print(secret.shape)
    secret = cv.resize(secret,(w,h))
    print(secret.shape)
    secret = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])(secret)  # 张量化
    secret = secret[np.newaxis, :, :, :]
    secret = secret.cuda()
    encode_S = Enet(secret).cuda()  # 生成的残差 输入的是secrets
    return encode_S




def Hide_Hnet_only(data, Enet_feat):
    """
    只计算Hide net,超快
    @param data:
    @param Enet_feat:
    @return:
    """
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    torch.cuda.empty_cache()
    Hnet = Hide_net.HideNet(input_nc=2,
                            output_nc=3, norm_layer=nn.BatchNorm2d,
                            output_function=nn.Tanh).cuda()
    model_path1 = model_path0.replace("Encode", "Hide")
    checkpoint1 = torch.load(model_path1, map_location=device)
    Hnet.load_state_dict(checkpoint1)
    Hnet.eval()
    data = transforms.Compose([
        transforms.ToTensor()
    ])(data)  # 张量化
    data = data[np.newaxis, :, :, :]
    data = data.cuda()
    res = Hnet(data, Enet_feat)  # 3通道
    out_contain = res + data
    out_image = tensor2np(out_contain)
    return out_image



def tensor2np(src):
    output = src.squeeze(0)
    output = np.array(output.cpu().detach().numpy(), dtype='float32')
    output = np.transpose(output, (1, 2, 0))
    output *= 255.0
    return output


def Reveal_one_pic(src, is_cuda=False):
    src = transforms.Compose([
        transforms.ToTensor()
    ])(src)  # 张量化
    src = src[np.newaxis, :, :, :]
    if is_cuda:
        src = src.type(torch.cuda.FloatTensor)
    else:
        src = src.type(torch.FloatTensor)
    print(src.shape)
    out_sectret = Reveal(src, is_cuda=is_cuda)
    cv.imwrite("test_picture/1.png", out_sectret * 255)
    return out_sectret


class Secret_message:
    def __init__(self):
        self.ip = self.get_ip()
        self.time = self.get_time()
        self.name = self.get_name()
        self.secret = self.get_secret()

    def get_ip(self):
        # 能用
        # ip = 169.254.240.224
        import socket
        # print(socket.gethostbyname(socket.getfqdn(socket.gethostname())))
        return str(socket.gethostbyname(socket.getfqdn(socket.gethostname())))

    def get_time(self):
        import datetime
        Time = datetime.datetime.now()
        # print("当前的日期和时间是 %s" % Time)
        # print("当前的年份是 %s" % Time.year)
        # print("当前的月份是 %s" % Time.month)

        year = str(Time.year)
        month = self.format_time(Time.month)
        day = self.format_time(Time.day)
        hour = self.format_time(Time.hour)
        minute = self.format_time(Time.minute)
        second = self.format_time(Time.second)

        # print("当前的日期是  %s" % Time.day)
        # print("当前小时是 %s" % Time.hour)
        # print("当前分钟是 %s" % Time.minute)
        # print("当前秒是  %s" % Time.second)
        Now_time = year + month + day + hour + minute + second
        return Now_time

    def format_time(self, ss):
        if len(str(ss)) < 2:
            ss = '0' + str(ss)
        else:
            ss = str(ss)
        return ss

    def get_name(self):
        getlogin_X = os.getlogin()
        return getlogin_X

    def get_secret(self):
        self.secret = self.name +"\n"+  self.ip +"\n" + self.time
        return self.secret


def test_hide_model():
    torch.cuda.empty_cache()
    path = r"test_picture/f1.png"
    data = cv.imread(path)
    data = data[0:512, 0:512]
    secret = "brighten-192.131.324.222-213123213"
    src = Hide_secret(data, secret)
    return src


def Hide_pic_port():
    """
    input:data   a numpy image  which has a size most modded by 16
    output:container a numpy image   which has the same size as the input_data
    """
    data = cv.imread(r"../data/DIV2K_train_HR/0801.png")
    data = data[0:h, 0:w]
    # data =255-data
    S = Secret_message()
    secret = S.secret  # 实时读取数据，主要为切换时间
    secret = "F1h3AlKw5\n445.342.796.448\n939073726462"
    torch.cuda.empty_cache()
    src = Hide_secret(data, secret)
    cv.imwrite("1.png",src)
    return src


from _datetime import datetime


def time_consume():
    start = datetime.now()
    end = datetime.now()
    print('用时：{:.4f}s'.format((end - start).seconds))


def test_reveal(path = r"C:\Users\brighten\Desktop\1.jpg"):
    torch.cuda.empty_cache()
    src = cv.imread(path)
    print(src.shape)
    src = src[::3, ::3]
    # src = src[:700,:]
    reve = Reveal_one_pic(src, is_cuda=False)
    cv.imwrite(r"C:\Users\brighten\Desktop\1_test.png", reve)
    # cv.imwrite( r"C:\Users\brighten\Desktop\data\5_\5_crop.jpg",src)
    cv.imshow('1', reve)
    cv.waitKey(0)


def test_hide():
    print("图像大小:{:d}*{:d}".format(h, h))
    a = Hide_pic_port()
    end = datetime.now()
    print("end,", end, "\nstartHnet:", start_Hnet, "\nstartEnet:", start_Enet)
    print('hnet用时：{:.10f}s'.format((end - start_Hnet).seconds))
    cv.imwrite("test_picture/test.png", a)


if __name__ == '__main__':
    Hide_pic_port()
