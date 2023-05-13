#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
2022/4/8 可以跑
2022/4/12  保险起见，设置batch_size为1，否则很多函数没办法使用，或者效果出错;主要面向jpeg_compress 形变

主要改动就是加入边缘，改变网络为3阶段网络，改变Secret为大字+小字的结合

Encode_net - Hide_net - Revealnet
S           C               Contain     out_S
"""

import argparse
import datetime
import random
import sys
import time
from os.path import join, isdir, isfile, abspath, dirname

import torch.optim as optim
import torch.utils.data.dataloader
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torchvision.utils import make_grid
import yaml

sys.path.append('../')
from JPEG import DiffJPEG
from Dataset.dataloader import Mydataset
from Functions.loss_functions import *
from Functions.utils import *
from lpips import lpips
from model_ldh import Revealnet, Hide_net, Encode_net

now_day = datetime.datetime.now().date()
now_day = str(now_day)[-5:]
# 命名格式为日期（月+日） +  train重要信息  根据日期命名runs文件
name = now_day + 'watermark_lunwen'
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=1, type=int, metavar='BT',
                    help='batch size')
parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--resume', default=[
    "/home/liu/sjw/save_model/watermark_base/BEST_Encode_07-08watermark_checkpoint35.pth",
    "/home/liu/sjw/save_model/watermark_base/BEST_Hide_07-08watermark_checkpoint35.pth",
    "/home/liu/sjw/save_model/watermark_base/BEST_Rev_07-08watermark_checkpoint35.pth"
    # "/home/liu/sjw/save_model/watermark_base/Encode_07-20watermark_add_noise_checkpoint215.pth",
    # "/home/liu/sjw/save_model/watermark_base/Hide_07-20watermark_add_noise_checkpoint215.pth",
    # "/home/liu/sjw/save_model/watermark_base/Rev_07-20watermark_add_noise_checkpoint215.pth"
], type=list, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--weight_decay', default=2e-2, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=3, type=int,
                    metavar='SS', help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.7, type=float,
                    help='learning rate decay parameter: Gamma')
parser.add_argument('--maxepoch', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--itersize', default=10, type=int,
                    metavar='IS', help='iter size')
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

with open('../cfg/setting.yaml', 'r') as f:
    Hp = yaml.load(f, Loader=yaml.SafeLoader)


def noise_layer(contain, OUT_SECRET,epoch = 0):
    """
    要求大多数情况下，contain 和 Secret进行的变换是等同的
    @param contain: 包含秘密信息 的cover
    @param OUT_SECRET: 秘密信息
    @return: 进行扰动之后的图像
    """
    # 此处必须先进行warp，再进行非几何变换操作。否则和实际不符合。拍摄时，先warp再相机jpeg压缩。或是jpeg->warp->jpeg
    # if random.randint(0, 10) % 3 == 1:
    #     a, b = random.randint(4, 12) / 10, random.randint(4, 12) / 10
    #     contain = F.interpolate(contain, scale_factor=(a, b),
    #                             mode='bilinear')  # 放缩函数，此处为缩小,放缩方法可选mode
    #     OUT_SECRET = F.interpolate(OUT_SECRET, scale_factor=(a, b), mode='bilinear')  # 对gt做同样操作
    # if random.randint(0, 9) % 3 == 1:
    #     # jpeg 双压缩
    #     a, b = contain.shape[2], contain.shape[3]
    #     q = random.randint(0, 10) % 4
    #     quality_list = [80, 85, 90, 95]  # 压缩质量不能太低
    #     quality = quality_list[q]
    #     J = DiffJPEG.DiffJPEG(width=b, differentiable=True, quality=quality, height=a)  # 压缩质量判定
    #     contain = J(contain)

    if random.randint(0, 9) % 3 == 1:
        # 扭曲
        t1 = random.randint(1, 5) / 10  # t从0.1-0.5都没有问题，可能会模糊
        t2 = random.randint(1, 5) / 10
        contain = warped(contain, t1, t2)
        OUT_SECRET = warped(OUT_SECRET, t1, t2)
    elif random.randint(0, 9) % 4 == 1:
        ang = random.randint(0, 360)
        contain = affine_rotating(contain, ang=ang)
        OUT_SECRET = affine_rotating(OUT_SECRET, ang=ang)
    elif random.randint(0, 9) % 4 == 1:
        contain = affine_big23(contain)
        OUT_SECRET = affine_big23(OUT_SECRET)
    elif random.randint(0, 9) % 4 == 2:
        contain = affine_small22(contain)
        OUT_SECRET = affine_small22(OUT_SECRET)
    elif random.randint(0, 9) % 4 == 1:
        contain = affine_small32(contain)
        OUT_SECRET = affine_small32(OUT_SECRET)
    elif random.randint(0, 9) % 4 == 1:
        contain = affine_small23(contain)
        OUT_SECRET = affine_small23(OUT_SECRET)

    if random.randint(0, 9) % 3 == 2:
        # gauss blur
        blur_layer = get_gaussian_kernel().cuda()
        contain = blur_layer(contain)
    if random.randint(0, 9) % 3 == 2:
        std_noise = (torch.rand(1) * 0.05).item()
        noise = torch.randn_like(contain) * std_noise
        contain = contain + noise

    #  加入jpeg压缩(可微)  必须进行概率判定，否则无法应对正常图像
    if random.randint(0, 9) % 3 == 1:
        a, b = contain.shape[2], contain.shape[3]
        q = random.randint(0, 10) % 4
        quality_list = [70, 80, 90, 95]  # 压缩质量不能太低
        quality = quality_list[q]
        J = DiffJPEG.DiffJPEG(width=b, differentiable=True, quality=quality, height=a)  # 压缩质量判定
        contain = J(contain)

    if random.randint(0, 2) % 3 == 1:
        # 加入光照和对比度变化
        brighten_layer = get_rnd_brightness(0.4, 0.3, args.batch_size).cuda()
        contain = contain + brighten_layer * 0.5
        # 放缩形变，放大缩小不变各占1/3，由于reveal为单纯卷积，不需要考虑整除问题,1-2倍的不恒等变换

    # noise layer结束
    return contain, OUT_SECRET


def noise_layer2(contain, OUT_SECRET, epoch):
    """
    需要体现样本均衡
    @param contain: contain
    @param OUT_SECRET: secret gt
    @param epoch: epoch
    @return: contain and secret gt after trans
    """
    if random.randint(0, 8) % 3 != 1:
        contain = transform_net(contain, Hp, global_step=epoch * 2)
    if random.randint(0, 10) % 3 == 1:
        contain = affine_big23(contain)
        OUT_SECRET = affine_big23(OUT_SECRET)
    elif random.randint(0, 10) % 3 == 1 and epoch > 3:
        # 扭曲
        t1 = random.randint(1, 5) / 10  # t从0.1-0.5都没有问题，可能会模糊
        t2 = random.randint(1, 5) / 10
        contain = warped(contain, t1, t2)
        OUT_SECRET = warped(OUT_SECRET, t1, t2)
    elif random.randint(0, 9) % 2 == 1:
        contain = affine_small22(contain)
        OUT_SECRET = affine_small22(OUT_SECRET)
    elif random.randint(0, 9) % 2 == 1:
        contain = affine_big22(contain)
        OUT_SECRET = affine_big22(OUT_SECRET)
    elif random.randint(0, 9) % 3 == 2:
        # 这个是不足以达到倍数缩放的攻击要求的
        a, b = random.randint(8, 12) / 10, random.randint(8, 12) / 10
        contain = F.interpolate(contain, scale_factor=(a, b),
                                mode='bilinear')  # 放缩函数，此处为缩小,放缩方法可选mode
        OUT_SECRET = F.interpolate(OUT_SECRET, scale_factor=(a, b), mode='bilinear')  # 对gt做同样操作
    return contain, OUT_SECRET


def noise_layer3(contain, OUT_SECRET, epoch, is_test=False):
    if is_test:
        batch_size = 1
    else:
        batch_size = args.batch_size
    t1 = random.randint(1, 5) / 10  # t从0.1-0.5都没有问题，可能会模糊
    t2 = random.randint(1, 5) / 10
    contain = warped(contain, t1, t2)
    OUT_SECRET = warped(OUT_SECRET, t1, t2)

    contain = transform_net(contain, Hp, global_step=epoch, batch_size=batch_size)
    return contain, OUT_SECRET


def noise_layer4(contain, OUT_SECRET, epoch=0, is_test=False):
    """
    要求大多数情况下，contain 和 Secret进行的变换是等同的
    @param contain: 包含秘密信息 的cover
    @param OUT_SECRET: 秘密信息
    @return: 进行扰动之后的图像
    """
    if is_test:
        batch_size = 1
    else:
        batch_size = args.batch_size
    if is_test == True or random.randint(0, 9) % 5 != 1:
        # 至少有1/4是没有噪声的，这个数据可能要调参。样本均衡!!!一定要样本均衡
        # if random.randint(0, 9) % 3 == 1:
        #     # jpeg 双压缩
        #     a, b = contain.shape[2], contain.shape[3]
        #     q = random.randint(0, 10) % 4
        #     quality_list = [80, 85, 90, 95]  # 压缩质量不能太低
        #     quality = quality_list[q]
        #     J = DiffJPEG.DiffJPEG(width=b, differentiable=True, quality=quality, height=a)  # 压缩质量判定
        #     contain = J(contain)
        a = random.randint(0, 22)
        if a % 3 == 1:
            # 扭曲
            t1 = random.randint(1, 5) / 10  # t从0.1-0.5都没有问题，可能会模糊
            t2 = random.randint(1, 5) / 10
            contain = warped(contain, t1, t2)
            OUT_SECRET = warped(OUT_SECRET, t1, t2)
        if a % 2 == 1:
            # 角度控制，不能太离谱
            ang = random.randint(-30, 30)
            contain = affine_rotating(contain, ang=ang)
            OUT_SECRET = affine_rotating(OUT_SECRET, ang=ang)
        if epoch < 30 and a % 3 == 2:
            # 随着epoch 增加，变化倍数增加，放大和扭曲是最重要的几何变化
            contain = affine_big22(contain)
            OUT_SECRET = affine_big22(OUT_SECRET)
        elif epoch < 30 and a % 4 == 2:
            contain = affine_big23(contain)
            OUT_SECRET = affine_big23(OUT_SECRET)
        if epoch > 30 and a % 3 == 2:
            contain = affine_big_epoch(contain, epoch)
            OUT_SECRET = affine_big_epoch(OUT_SECRET, epoch)

        if random.randint(0, 9) % 3 == 2:
            # gauss blur
            blur_layer = get_gaussian_kernel().cuda()
            contain = blur_layer(contain)
        if random.randint(0, 9) % 3 == 2:
            std_noise = (torch.rand(1) * 0.05).item()
            noise = torch.randn_like(contain) * std_noise
            contain = contain + noise

        if random.randint(0, 9) % 3 != 1:
            a, b = contain.shape[2], contain.shape[3]
            q = random.randint(0, 10) % 4
            quality_list = [80, 85, 90, 95]  # 压缩质量不能太低
            quality = quality_list[q]
            J = DiffJPEG.DiffJPEG(width=b, differentiable=True, quality=quality, height=a)  # 压缩质量判定
            contain = J(contain)

        if random.randint(0, 2) % 2 == 1:
            # 加入光照和对比度变化
            a = 0.1 + epoch / 100
            theta = np.min([a, 1.0])
            brighten_layer = get_rnd_brightness(0.4, 0.3, batch_size).cuda()
            contain = contain + brighten_layer * theta

    # noise layer结束
    return contain, OUT_SECRET


def noise_layer5(contain, OUT_SECRET, epoch=0, is_test=False):
    if is_test:
        batch_size = 1
    else:
        batch_size = args.batch_size
    if is_test == True or random.randint(0, 9) % 4 != 1:
        if random.randint(0, 9) % 3 == 1:
            width = 256
            rnd_trans = 0.1
            rnd_trans_ramp = 1000
            global_step = epoch * 8
            rnd_tran = min(rnd_trans * global_step / rnd_trans_ramp, rnd_trans)
            rnd_tran = np.random.uniform() * rnd_tran
            Ms = get_rand_transform_matrix(width, np.floor(width * rnd_tran), batch_size=batch_size).cuda()
            contain = warp_genel(contain, Ms)
            OUT_SECRET = warp_genel(OUT_SECRET, Ms)
        elif random.randint(0, 9) % 4 == 1:
            ang = random.randint(-36, 36)
            contain = affine_rotating(contain, ang=ang)
            OUT_SECRET = affine_rotating(OUT_SECRET, ang=ang)
        elif random.randint(0, 9) % 4 == 1:
            contain = affine_big23(contain)
            OUT_SECRET = affine_big23(OUT_SECRET)
        elif random.randint(0, 9) % 4 == 2:
            contain = affine_small22(contain)
            OUT_SECRET = affine_small22(OUT_SECRET)
        elif random.randint(0, 9) % 4 == 1:
            contain = affine_small32(contain)
            OUT_SECRET = affine_small32(OUT_SECRET)
        elif random.randint(0, 9) % 4 == 1:
            contain = affine_small23(contain)
            OUT_SECRET = affine_small23(OUT_SECRET)

        if epoch > 30 and random.randint(0, 9) % 3 == 2:
            contain = affine_big_epoch(contain, epoch)
            OUT_SECRET = affine_big_epoch(OUT_SECRET, epoch)
        if random.randint(0, 9) % 4 != 2:
            contain = transform_net(contain, Hp, global_step=epoch * 2)
    return contain, OUT_SECRET


def adjust_learning_rate(optimizer, epoch):
    # 调节器  抄的UDP代码，在每个epoch开始时作用
    # 无视Adam优化器，强制要求lr随epoch 变化，每隔10个epoch下降10倍的lr
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.001 * (0.2 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def is_bestpth(now, epoch):
    # loss越小越好
    global BEST
    if now < BEST:
        # if epoch > 5:
        BEST = now
        return True
    else:
        return False


def main():
    device = 'xxl'
    args.cuda = True
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    trainData = Mydataset(device=device, train_val_test_mode='train')
    valData = Mydataset(device=device, train_val_test_mode='val')
    testData = Mydataset(device=device, train_val_test_mode='val')

    trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=args.batch_size, num_workers=1)
    valDataLoader = torch.utils.data.DataLoader(dataset=valData, batch_size=args.batch_size, num_workers=4,
                                                shuffle=True)
    testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=1, num_workers=0)

    # model_udh  此处固定为一张图嵌入一张图  Hnet return a same type of picture as the input
    Enet = Encode_net.UnetGenerator(input_nc=1,
                                    output_nc=1, num_downs=5, norm_layer=nn.BatchNorm2d,
                                    output_function=nn.Sigmoid)
    Hnet = Hide_net.HideNet(input_nc=2,
                            output_nc=3, norm_layer=nn.BatchNorm2d,
                            output_function=nn.Tanh)
    Rnet = Revealnet.RevealNet(input_nc=3,
                               output_nc=1, norm_layer=nn.BatchNorm2d,
                               output_function=nn.Sigmoid)

    if torch.cuda.is_available():
        Enet.cuda()
        Hnet.cuda()
        Rnet.cuda()
    else:
        Enet.cpu()
        Hnet.cpu()
        Rnet.cpu()

    # 可以加 权重初始化
    # Hnet.apply(weights_init)
    # Rnet.apply(weights_init)

    params = list(Enet.parameters()) + list(Hnet.parameters()) + list(Rnet.parameters())  # 对两个模型同时进行优化
    optimizer = optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)  # 优化器只要一个就行
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    # 加载模型 根据args.resume  加载权重
    if isfile(args.resume[0]) and isfile(args.resume[1] and isfile(args.resume[2])):
        print("=> loading checkpoint '{}'".format(args.resume))
        # 从这个参数中读取权重的路径
        checkpoint0 = torch.load(args.resume[0])
        checkpoint1 = torch.load(args.resume[1])
        checkpoint2 = torch.load(args.resume[2])
        Enet.load_state_dict(checkpoint0['state_dict'])
        Hnet.load_state_dict(checkpoint1['state_dict'])
        Rnet.load_state_dict(checkpoint2['state_dict'])
        print("=> loaded checkpoint '{}'".format(args.resume))
    else:
        print("=> !!!!!!! checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.start_epoch, args.maxepoch):
        print("------------epoch-----------", epoch)
        adjust_learning_rate(optimizer, epoch)  # 调节器 去掉，这个调节器过于暴力

        train_avg = train(Enet=Enet, Hnet=Hnet, Rnet=Rnet, optimizer=optimizer,
                          dataParser=trainDataLoader, epoch=epoch)
        val_avg = val(Enet=Enet, Hnet=Hnet, Rnet=Rnet, dataParser=valDataLoader, epoch=epoch)
        test(Enet=Enet, Hnet=Hnet, Rnet=Rnet, dataParser=testDataLoader, epoch=epoch)  # test直接用val就行

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
        output_name0 = output_name_file_name % (epoch)
        output_name1 = output_name_file_name % (epoch)
        output_name2 = output_name_file_name % (epoch)

        if is_bestpth(val_avg['loss_avg'], epoch):
            # 保存三阶段网络参数
            save_model_name_stage0 = os.path.join(args.model_save_dir, 'BEST_Encode_' + output_name0)
            save_model_name_stage1 = os.path.join(args.model_save_dir, 'BEST_Hide_' + output_name1)
            save_model_name_stage2 = os.path.join(args.model_save_dir, 'BEST_Rev_' + output_name2)

            torch.save(Enet.state_dict(), save_model_name_stage0)
            torch.save(Hnet.state_dict(), save_model_name_stage1)
            torch.save(Rnet.state_dict(), save_model_name_stage2)
        elif epoch % 5 == 0:
            save_model_name_stage0 = os.path.join(args.model_save_dir, 'Encode_' + output_name0)
            save_model_name_stage1 = os.path.join(args.model_save_dir, 'Hide_' + output_name1)
            save_model_name_stage2 = os.path.join(args.model_save_dir, 'Rev_' + output_name2)

            torch.save(Enet, save_model_name_stage0)
            torch.save(Hnet, save_model_name_stage1)
            torch.save(Rnet, save_model_name_stage2)
    print('训练已完成!')


def train(Enet, Hnet, Rnet, optimizer, dataParser, epoch):
    # 读取数据的迭代器
    train_epoch = len(dataParser)
    batch_time = Averagvalue()
    data_time = Averagvalue()
    Lossall = Averagvalue()

    Enet.train()
    Hnet.train()
    Rnet.train()

    end = time.time()
    for batch_index, input_data in enumerate(dataParser):
        # 读取数据的时间
        data_time.update(time.time() - end)
        secrets_in = input_data['secret_in'].cuda()
        cover = input_data['contain'].cuda()
        OUT_SECRET = input_data['secret_out'].cuda()  # 没有进行归一化的secret，仅仅作为gt使用
        with torch.set_grad_enabled(True):
            secrets_in.requires_grad = True
            cover.require_grad = True

            optimizer.zero_grad()
            encode_S = Enet(secrets_in).cuda()  # 生成的残差 输入的是secrets
            # contain = (cover + Hide_out).cuda()  # 获取含有S的Cover ，原本为和残差相加的contain

            res = Hnet(cover, encode_S)  # 3通道

            contain = cover + res

            # rgb_loss = rgb_diff_loss(res)

            # 计算loss和noise layer无关，在noise layer之前
            loss_fn = lpips.LPIPS(net='alex').cuda()
            loss_lpips = loss_fn(cover, contain)  # 输入和输出的cover进行判定  使用lpips函数，判定图像的相似度,越接近越小
            loss_lpips_mean = torch.mean(loss_lpips)  # 防止输出高维向量

            ssim_loss = SSIM()
            ssim = ssim_loss(contain, cover)

            # 颜色角loss
            # cosine_loss = cosine_distance(contain, cover)

            # color_anger = color_Loss()
            # color_loss = color_anger(cover, contain)

            loss_l2 = nn.MSELoss()
            loss_res = loss_l2(cover, contain)  # contain和contain_out的结果要接近，实际上和原本没有区别
            # 对于水印嵌入网络的loss计算已完成，可以进行contain的任意操作
            # if epoch > 3:
            #     如果加入预训练模型，可以取消
            #     此处很重要，相等于预训练模型的作用，noise layer不能一开始就强度很高,否则造成水印过于明显的问题
            # if epoch >3 :
            # OUT_SECRET 只应该进行集合变化
            contain, OUT_SECRET = noise_layer(contain, OUT_SECRET, epoch)

            secret_out = Rnet(contain)  # recover

            ent_losss = entropy_loss(secret_out)

            # secret_out = torch.where(secret_out > 0.5, torch.tensor(1).cuda(), torch.tensor(0).cuda())

            loss_o_sec = cross_entropy_loss(secret_out, OUT_SECRET)  # loss2 输出的sec和输入的sec
            # ssim2 = 1 - ssim_loss(secret_out, OUT_SECRET)
            Loss_all = loss_o_sec * 1 + loss_lpips_mean * 0.1 + 0.1 * loss_res + (
                    1 - ssim) * 0.5 + 0.1 * ent_losss  # + ssim2 * 0.1  # + color_loss * 5  # 最终的loss

            # if epoch > 20:
            #     Loss_all += rgb_loss * 0.2

            writer.add_scalars('loss_gather', {'all': Loss_all.item(),
                                               'lpips': loss_lpips_mean.item(),
                                               'loss_l2': loss_res.item(),
                                               'ent_losss': ent_losss,
                                               'loss_out_secret': loss_o_sec.item(),
                                               'ssim': (1 - ssim) * 0.5,
                                               # 'ssim_secret': ssim2*0.1
                                               # 'color_loss': color_loss * 5
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


def val(Enet, Hnet, Rnet, dataParser, epoch):
    # val 模块不加入扰动
    train_epoch = len(dataParser)
    batch_time = Averagvalue()
    data_time = Averagvalue()
    Lossall = Averagvalue()

    Enet.eval()
    Hnet.eval()
    Rnet.eval()

    end = time.time()
    for batch_index, input_data in enumerate(dataParser):
        data_time.update(time.time() - end)
        secrets_in = input_data['secret_in'].cuda()
        cover = input_data['contain'].cuda()
        img = input_data['image'].cuda()  # 有归一化的
        OUT_SECRET = input_data['secret_out'].cuda()  # 没有进行归一化的secret，仅仅作为gt使用

        with torch.set_grad_enabled(False):
            secrets_in.requires_grad = False
            encode_S = Enet(secrets_in).cuda()  # 生成的残差 输入的是secrets
            res = Hnet(img, encode_S)  # 3通道
            contain = cover + res

            # rgb_loss = rgb_diff_loss(res)
            color_anger = color_Loss()
            color_loss = color_anger(cover, contain)

            loss_fn = lpips.LPIPS(net='alex').cuda()
            loss_lpips = loss_fn(cover, contain)  # 输入和输出的cover进行判定  使用lpips函数，判定图像的相似度,越接近越小
            loss_lpips_mean = torch.mean(loss_lpips)  # 防止输出高维向量

            # 去掉l2 loss
            loss_l2 = nn.MSELoss()
            loss_res = loss_l2(cover, contain)  # contain和contain_out的结果要接近，实际上和原本没有区别

            ssim_loss = SSIM()
            ssim = ssim_loss(contain, cover)

            # 对于水印嵌入网络的loss计算已完成，可以进行contain的任意操作
            secret_out = Rnet(contain)  # recover

            ent_losss = entropy_loss(secret_out)
            # secret_out = torch.where(secret_out > 0.5, torch.tensor(1).cuda(), torch.tensor(0).cuda())
            loss_o_sec = cross_entropy_loss(secret_out, OUT_SECRET)  # loss2 输出的sec和输入的sec

            # 颜色角loss

            Loss_all = loss_o_sec * 1 + loss_lpips_mean * 0.1 + 0.1 * loss_res + (
                    1 - ssim) * 0.5 + 0.1 * ent_losss  # 最终的loss

            # if epoch > 20:
            #     Loss_all += rgb_loss * 0.2

            writer.add_scalars('loss_val', {'all': Loss_all.item(),
                                            'lpips': loss_lpips_mean.item(),
                                            'loss_l2': loss_res.item(),
                                            'ent_losss': ent_losss,
                                            'loss_out_secret': loss_o_sec.item(),
                                            'ssim': (1 - ssim) * 0.5
                                            # 'color_loss': color_loss * 5
                                            }, global_step=epoch * train_epoch + batch_index)
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


@torch.no_grad()
def test(Enet, Hnet, Rnet, dataParser, epoch):  # 测试集

    Enet.eval()
    Hnet.eval()
    Rnet.eval()
    for batch_index, input_data in enumerate(dataParser):
        if batch_index % 8 == 0:
            images = input_data['image'].cuda()
            secrets_in = input_data['secret_in'].cuda()
            cover = input_data['contain'].cuda()
            img = input_data['image'].cuda()  # 有归一化的
            with torch.set_grad_enabled(False):
                images.requires_grad = False
                secrets_in.requires_grad = False
                encode_S = Enet(secrets_in).cuda()  # 生成的残差 输入的是secrets
                res = Hnet(img, encode_S)  # 3通道
                contain = cover + res

                contain_origin = contain

                secret_out_without = Rnet(contain)
                # secret_out_without = torch.where(secret_out_without > 0.5, torch.tensor(1).cuda(), torch.tensor(0).cuda())

                contain, _ = noise_layer(contain, secrets_in, epoch)
                secret_out_withwarp = Rnet(contain)  # recover

                writer.add_image('contain:%d' % batch_index,
                                 make_grid(contain, nrow=2), global_step=epoch)
                writer.add_image('contain_origin:%d' % batch_index,
                                 make_grid(contain_origin, nrow=2), global_step=epoch)
                writer.add_image('secret_out_without:%d' % batch_index,
                                 make_grid(secret_out_without, nrow=2), global_step=epoch)
                writer.add_image('secret_out_withwarp:%d' % batch_index,
                                 make_grid(secret_out_withwarp, nrow=2), global_step=epoch)
                writer.add_image('res:%d' % batch_index,
                                 make_grid(res, nrow=2), global_step=epoch)


if __name__ == '__main__':
    main()
