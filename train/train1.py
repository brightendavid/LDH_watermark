#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
2022/4/8 可以跑
2022/4/12  保险起见，设置batch_size为1，否则很多函数没办法使用，或者效果出错;主要面向jpeg_compress 形变
"""

import argparse
import datetime
import random

import time
from os.path import join, isdir, isfile, abspath, dirname
import sys

sys.path.append('../')

from JPEG import DiffJPEG
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torchvision.utils import make_grid

from Dataset.dataloader import Mydataset
from Functions.loss_functions import *
from Functions.utils import *
from lpips import lpips
from model_udh import Revealnet, Hide

now_day = datetime.datetime.now().date()
now_day = str(now_day)[-5:]
# 命名格式为日期（月+日） +  train重要信息  根据日期命名runs文件
name = now_day + 'watermark'
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=1, type=int, metavar='BT',
                    help='batch size')
parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--resume', default=[
    "",
    ""
], type=list, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--weight_decay', default=2e-2, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=3, type=int,
                    metavar='SS', help='learning rate step size')
# todo 调节gamma
parser.add_argument('--gamma', '--gm', default=0.2, type=float,
                    help='learning rate decay parameter: Gamma')
parser.add_argument('--maxepoch', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--itersize', default=10, type=int,
                    metavar='IS', help='iter size')
# =============== misc
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU ID')
parser.add_argument('--model_save_dir', type=str, help='model_save_dir',
                    default='../save_model/' + name)
parser.add_argument('--per_epoch_freq', type=int, help='per_epoch_freq', default=50)

parser.add_argument('--fuse_loss_weight', type=int, help='fuse_loss_weight', default=12)
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
model_save_dir = abspath(dirname(__file__))
model_save_dir = join(model_save_dir, args.model_save_dir)

if not isdir(model_save_dir):
    os.makedirs(model_save_dir)

# tensorboard 使用
writer = SummaryWriter(
    '../runs/' + name)
email_header = 'Python'
output_name_file_name = name + '_checkpoint%d.pth'

BEST = 10


def noise_layer(contain_out, OUT_SECRET):
    # noise layer 开始
    # 对transform_model进行扰动 加噪声 对contain 加噪声，不能对生成的sec加噪声 add noise
    std_noise = (torch.rand(1) * 0.05).item()
    noise_layer = torch.randn_like(contain_out) * std_noise
    contain_out = contain_out + noise_layer

    #  加入jpeg压缩(可微)  必须进行概率判定，否则无法应对正常图像
    if random.randint(0, 9) % 3 != 2:
        a, b = contain_out.shape[2], contain_out.shape[3]
        q = random.randint(0, 10) % 4
        quality_list = [60, 70, 80, 90]  # 压缩质量不能太低
        quality = quality_list[q]
        J = DiffJPEG.DiffJPEG(width=b, differentiable=True, quality=quality, height=a)  # 压缩质量判定
        contain_out = J(contain_out)

    if random.randint(0, 20) % 2 == 1:
        # 加入光照和对比度变化
        brighten_layer = get_rnd_brightness(0.4, 0.3, args.batch_size).cuda()
        contain_out = contain_out + brighten_layer
        # 放缩形变，放大缩小不变各占1/3，由于reveal为单纯卷积，不需要考虑整除问题,1-2倍的不恒等变换
        if random.randint(0, 10) % 2 == 1:
            a, b = random.randint(4, 12) / 10, random.randint(4, 12) / 10
            contain_out = F.interpolate(contain_out, scale_factor=(a, b),
                                        mode='bilinear')  # 放缩函数，此处为缩小,放缩方法可选mode
            OUT_SECRET = F.interpolate(OUT_SECRET, scale_factor=(a, b), mode='bilinear')  # 对gt做同样操作

    # 仿射变化
    if random.randint(0, 9) % 2 == 1:
        if random.randint(0, 9) % 3 == 1:
            # affine changes a lot
            # rotate
            ang = random.randint(0, 360)
            contain_out = affine_rotating(contain_out, ang=ang)
            OUT_SECRET = affine_rotating(OUT_SECRET, ang=ang)
        if random.randint(0, 9) % 3 == 1:
            contain_out = affine_big23(contain_out)
            OUT_SECRET = affine_big23(OUT_SECRET)
        if random.randint(0, 9) % 3 == 2:
            contain_out = affine_big22(contain_out)
            OUT_SECRET = affine_big22(OUT_SECRET)
        if random.randint(0, 9) % 3 == 2:
            contain_out = affine_big50(contain_out)
            OUT_SECRET = affine_big50(OUT_SECRET)
        if random.randint(0, 9) % 3 == 2:
            contain_out = affine_small22(contain_out)
            OUT_SECRET = affine_small22(OUT_SECRET)
        if random.randint(0, 9) % 4 == 2:
            contain_out = affine_small32(contain_out)
            OUT_SECRET = affine_small32(OUT_SECRET)
        if random.randint(0, 9) % 4 == 3:
            contain_out = affine_small23(contain_out)
            OUT_SECRET = affine_small23(OUT_SECRET)
    # noise layer结束
    return contain_out, OUT_SECRET


def adjust_learning_rate(optimizer, epoch):
    # 调节器  抄的UDP代码，在每个epoch开始时作用
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.001 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def is_bestpth(now):
    # loss越小越好
    global BEST
    if now < BEST:
        BEST = now
        return True
    else:
        return False


def main():
    device = 'xxl'
    args.cuda = True
    torch.cuda.empty_cache()
    trainData = Mydataset(device=device, train_val_test_mode='train')
    valData = Mydataset(device=device, train_val_test_mode='val')
    testData = Mydataset(device=device, train_val_test_mode='val')

    trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=args.batch_size, num_workers=1)
    valDataLoader = torch.utils.data.DataLoader(dataset=valData, batch_size=args.batch_size, num_workers=4,
                                                shuffle=True)
    testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=1, num_workers=0)

    # model_udh  此处固定为一张图嵌入一张图  Hnet return a same type of picture as the input
    Hnet = Hide.UnetGenerator(input_nc=1,
                              output_nc=3 * 1, num_downs=5, norm_layer=nn.BatchNorm2d,
                              output_function=nn.Sigmoid)
    # Rnet return a (1,h,w) picture
    Rnet = Revealnet.RevealNet(input_nc=3,
                               output_nc=1, norm_layer=nn.BatchNorm2d,
                               output_function=nn.Sigmoid)

    if torch.cuda.is_available():
        Hnet.cuda()
        Rnet.cuda()
    else:
        Hnet.cpu()
        Rnet.cpu()
    # 可以加 权重初始化
    # Hnet.apply(weights_init)
    # Rnet.apply(weights_init)

    params = list(Hnet.parameters()) + list(Rnet.parameters())  # 对两个模型同时进行优化
    optimizer = optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)  # 优化器只要一个就行
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    # 加载模型 根据args.resume  加载权重
    if isfile(args.resume[0]) and isfile(args.resume[1]):
        print("=> loading checkpoint '{}'".format(args.resume))
        # 从这个参数中读取权重的路径
        checkpoint1 = torch.load(args.resume[0])
        checkpoint2 = torch.load(args.resume[1])
        Hnet.load_state_dict(checkpoint1['state_dict'])
        Rnet.load_state_dict(checkpoint2['state_dict'])
        print("=> loaded checkpoint '{}'".format(args.resume))
    elif isfile(args.resume[0]) and not isfile(args.resume[1]):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint1 = torch.load(args.resume[0])
        Hnet.load_state_dict(checkpoint1['state_dict'])
        print("=> loaded checkpoint '{}'".format(args.resume))
    elif not isfile(args.resume[0]) and isfile(args.resume[1]):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint2 = torch.load(args.resume[1])
        Rnet.load_state_dict(checkpoint2['state_dict'])
        print("=> loaded checkpoint '{}'".format(args.resume))
    else:
        print("=> !!!!!!! checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.start_epoch, args.maxepoch):
        print("------------epoch-----------", epoch)
        adjust_learning_rate(optimizer, epoch)  # 调节器

        train_avg = train(Hnet=Hnet, Rnet=Rnet, optimizer=optimizer,
                          dataParser=trainDataLoader, epoch=epoch)
        val_avg = val(Hnet=Hnet, Rnet=Rnet, dataParser=valDataLoader, epoch=epoch)
        test(Hnet=Hnet, Rnet=Rnet, dataParser=testDataLoader, epoch=epoch)  # test直接用val就行
        """"""""""""""""""""""""""""""
        "          写入图             "
        """"""""""""""""""""""""""""""
        try:
            writer.add_scalars('lr_per_epoch', {'stage1': scheduler.get_lr(),
                                                }, global_step=epoch)
            writer.add_scalars('tr-val_avg_loss_per_epoch', {'train': train_avg['loss_avg'],
                                                             'val': val_avg['loss_avg'],
                                                             }, global_step=epoch)
        except Exception as e:
            print(e)
        scheduler.step(epoch=epoch)
        # 保存val表现结果最好的权重
        output_name1 = output_name_file_name % \
                       (epoch)
        output_name2 = output_name_file_name % \
                       (epoch)

        if is_bestpth(val_avg['loss_avg']):
            save_model_name_stage1 = os.path.join(args.model_save_dir, 'Hide_' + output_name1)
            save_model_name_stage2 = os.path.join(args.model_save_dir, 'Rev_' + output_name2)
            torch.save({'epoch': epoch, 'state_dict': Hnet.state_dict(), 'optimizer': optimizer.state_dict()},
                       save_model_name_stage1)
            torch.save({'epoch': epoch, 'state_dict': Rnet.state_dict(), 'optimizer': optimizer.state_dict()},
                       save_model_name_stage2)
    print('训练已完成!')


def train(Hnet, Rnet, optimizer, dataParser, epoch):
    # 读取数据的迭代器

    train_epoch = len(dataParser)
    batch_time = Averagvalue()
    data_time = Averagvalue()
    Lossall = Averagvalue()

    Hnet.train()
    Rnet.train()

    end = time.time()
    for batch_index, input_data in enumerate(dataParser):
        # 读取数据的时间
        data_time.update(time.time() - end)
        images = input_data['image'].cuda()
        secrets_in = input_data['secret_in'].cuda()
        contains = input_data['contain'].cuda()
        OUT_SECRET = input_data['secret_out'].cuda()  # 没有进行归一化的secret，仅仅作为gt使用

        with torch.set_grad_enabled(True):
            secrets_in.requires_grad = True
            optimizer.zero_grad()
            Hide_out = Hnet(secrets_in).cuda() # 生成的残差 输入的是secrets
            contain_out = (contains + Hide_out).cuda()  # 获取含有S的Cover

            # 计算loss和noise layer无关，在noise layer之前
            loss_fn = lpips.LPIPS(net='alex').cuda()
            loss_lpips = loss_fn(contains, contain_out)  # 输入和输出的cover进行判定  使用lpips函数，判定图像的相似度,越接近越小
            loss_lpips_mean = torch.mean(loss_lpips)  # 防止输出高维向量
            loss_l2 = nn.MSELoss()
            loss_res = loss_l2(contains, contain_out)  # contain和contain_out的结果要接近，实际上和原本没有区别
            # 对于水印嵌入网络的loss计算已完成，可以进行contain的任意操作
            contain_out, OUT_SECRET = noise_layer(contain_out, OUT_SECRET)

            secret_out = Rnet(contain_out)  # recover

            loss_o_sec = cross_entropy_loss(secret_out, OUT_SECRET)  # loss2 输出的sec和输入的sec
            ent_losss = entropy_loss(secret_out)
            Loss_all = loss_o_sec + loss_lpips_mean * 0.5 + loss_res * 0.1 + ent_losss * 0.2  # 最终的loss
            # OUT secret下降很快，说明提取水印相对容易
            print("loss_o_sec----------", loss_o_sec)
            print(loss_lpips_mean)
            print(loss_res)
            print(ent_losss)
            writer.add_scalars('loss_gather', {'all': Loss_all.item(),
                                               'lpips': loss_lpips_mean.item(),
                                               'loss_l2': loss_res.item(),
                                               'ent_losss': ent_losss,
                                               'loss_out_secret': loss_o_sec.item(),
                                               }, global_step=epoch * train_epoch + batch_index)
            Loss_all.backward()
            optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        Lossall.update(Loss_all.item())  # update the data in
        if batch_index % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, train_epoch) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=Lossall)
            print(info)
        if batch_index >= train_epoch:
            break
    return {'loss_avg': Lossall.avg}


def val(Hnet, Rnet, dataParser, epoch):
    # val 模块不加入扰动
    train_epoch = len(dataParser)
    batch_time = Averagvalue()
    data_time = Averagvalue()
    Lossall = Averagvalue()
    Hnet.eval()
    Rnet.eval()
    end = time.time()
    for batch_index, input_data in enumerate(dataParser):
        # 读取数据的时间
        data_time.update(time.time() - end)
        print("batch_index", batch_index)
        images = input_data['image'].cuda()
        secrets_in = input_data['secret_in'].cuda()
        contains = input_data['contain'].cuda()
        OUT_SECRET = input_data['secret_out'].cuda()  # 没有归一化的Secret
        with torch.set_grad_enabled(False):
            contains.requires_grad = False
            Hide_out = Hnet(secrets_in).cuda()  # 生成的残差
            contain_out = (contains + Hide_out).cuda()  # 获取含有S的Cover
            # 对transform_model进行扰动 加噪声 对contain 加噪声，不能对生成的sec加噪声 add noise
            loss_fn = lpips.LPIPS(net='alex').cuda()
            loss_lpips = loss_fn(contains, contain_out)  # 输入和输出的cover进行判定  使用lpips函数，判定图像的相似度,越接近越小
            loss_lpips_mean = torch.mean(loss_lpips)
            loss_l2 = nn.MSELoss()
            loss_res = loss_l2(contains, contain_out)  # contain和contain_out的结果要接近，实际上和原本没有区别

            contain_out, OUT_SECRET = noise_layer(contain_out, OUT_SECRET)

            secret_out = Rnet(contain_out)  # recover
            loss_o_sec = cross_entropy_loss(secret_out, OUT_SECRET)  # loss2 输出的sec和输入的sec
            ent_losss = entropy_loss(secret_out)
            Loss_all = loss_o_sec + loss_lpips_mean * 0.5 + loss_res * 0.1 + ent_losss * 0.2
        writer.add_scalars('loss_gather', {'all': Loss_all.item(),
                                           'lpips': loss_lpips_mean.item(),
                                           'loss_l2': loss_res.item(),
                                           'loss_out_secret': loss_o_sec.item(),
                                           }, global_step=epoch * train_epoch + batch_index)
        batch_time.update(time.time() - end)
        end = time.time()
        Lossall.update(Loss_all.item())  # update the data in Averagvalue
        if batch_index % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, batch_index, train_epoch) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=Lossall)
            print(info)
        if batch_index >= train_epoch:
            break
    return {'loss_avg': Lossall.avg
            }


@torch.no_grad()
def test(Hnet, Rnet, dataParser, epoch):  # 测试集
    Hnet.eval()
    Rnet.eval()
    for batch_index, input_data in enumerate(dataParser):
        if batch_index > 6:
            break
        images = input_data['image'].cuda()
        secrets_in = input_data['secret_in'].cuda()
        contains = input_data['contain'].cuda()
        with torch.set_grad_enabled(False):
            images.requires_grad = False
            res = Hnet(secrets_in)
            container = contains + res
            out_sec = Rnet(container)
            writer.add_image('images:%d' % (batch_index),
                             make_grid(images, nrow=2), global_step=epoch)
            writer.add_image('container:%d' % (batch_index),
                             make_grid(container, nrow=2), global_step=epoch)
            writer.add_image('out_secrect:%d' % (batch_index),
                             make_grid(out_sec, nrow=2), global_step=epoch)


if __name__ == '__main__':
    main()
