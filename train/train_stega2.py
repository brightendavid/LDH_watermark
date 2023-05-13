#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
对Stega Stamp水印算法的魔改，实际上很多都是不需要的

使用的网络框架都是差不多的Unet+多层卷积 的Encode-Decode框架，本算法在Decode网络之前中使用STN模型，空间转换网络用于调整空间变化
这个STN框架需要encode和decode网络实现稳定，之后再加入


主要就是测试算法的水印形式是字符01序列的必要性
使用的loss和数据集都不是和原本的完全一样

Discriminator就是一个判别器，和lpips loss实际上差不多，7/2版本没有加入，但是这个结构还是保留的;
noise layer 使用的是stegastamp版本，自创的给删除了.

detect模块，就不加了，需要矫正，说到底还是不行的
网络输入为cover和01序列，01序列有dense 之后进行reshape,转为
"""
import argparse
import datetime
import sys
import time
from os.path import join, isdir, isfile, abspath, dirname
sys.path.append('../')
import torch.optim as optim
import torch.utils.data.dataloader
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torchvision.utils import make_grid
from Dataset.dataloader_bit import Bitdataset
from Functions.loss_functions import *
from Functions.utils import *
from lpips import lpips
from model_stega import stega_net
import torch
now_day = datetime.datetime.now().date()
now_day = str(now_day)[-5:]
# 命名格式为日期（月+日） +  train重要信息  根据日期命名runs文件
name = now_day + 'stegastamp'
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=4, type=int, metavar='BT',
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


def adjust_learning_rate(optimizer, epoch):
    # 调节器  抄的UDP代码，在每个epoch开始时作用
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.001 * (0.1 ** (epoch // 20))
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

def noise_layer(contain,is_test = False):
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
    if random.randint(0, 9) % 3 == 1:
        # 扭曲
        t1 = random.randint(1, 5) / 10  # t从0.1-0.5都没有问题，可能会模糊
        t2 = random.randint(1, 5) / 10
        contain = warped(contain, t1, t2)
    elif random.randint(0, 9) % 4 == 1:
        contain = affine_big23(contain)
    elif random.randint(0, 9) % 4 == 1:
        ang = random.randint(0, 360)
        contain = affine_rotating(contain, ang=ang)
    elif random.randint(0, 9) % 4 == 2:
        contain = affine_small22(contain)
    elif random.randint(0, 9) % 4 == 1:
        contain = affine_small32(contain)
    elif random.randint(0, 9) % 4 == 1:
        contain = affine_small23(contain)
    if random.randint(0, 9) % 3 == 2:
        # gauss blur
        blur_layer = get_gaussian_kernel().cuda()
        contain = blur_layer(contain)
    if random.randint(0, 9) % 3 == 2:
        std_noise = (torch.rand(1) * 0.05).item()
        noise = torch.randn_like(contain) * std_noise
        contain = contain + noise
        contain = torch.clamp(contain, 0, 1)  # 给上下限
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
        brighten_layer = get_rnd_brightness(0.4, 0.3, batch_size).cuda()
        contain = contain + brighten_layer * 0.5
        contain = torch.clamp(contain, 0, 1)
        # 放缩形变，放大缩小不变各占1/3，由于reveal为单纯卷积，不需要考虑整除问题,1-2倍的不恒等变换
    # noise layer结束
    return contain

def main():
    device = 'xxl'
    args.cuda = True
    torch.cuda.empty_cache()
    trainData = Bitdataset(device=device, train_val_test_mode='train')
    valData = Bitdataset(device=device, train_val_test_mode='val')
    testData = Bitdataset(device=device, train_val_test_mode='val')

    trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=args.batch_size, num_workers=1)
    valDataLoader = torch.utils.data.DataLoader(dataset=valData, batch_size=args.batch_size, num_workers=4,
                                                shuffle=True)
    testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=1, num_workers=0)
    # model_udh  此处固定为一张图嵌入一张图  Hnet return a same type of picture as the input
    Encoder = stega_net.StegaStampEncoder().to("cuda")  # encode
    Decoder = stega_net.StegaStampDecoder().to("cuda")  # decode
    g_vars = list(Encoder.parameters()) + list(Decoder.parameters())
    optimizer_g = optim.Adam(g_vars, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)  # 优化器只要一个就行
    scheduler = lr_scheduler.StepLR(optimizer_g, step_size=args.stepsize, gamma=args.gamma)
    # 加载模型 根据args.resume  加载权重
    if isfile(args.resume[0]) and isfile(args.resume[1]):
        print("=> loading checkpoint '{}'".format(args.resume))
        # 从这个参数中读取权重的路径
        checkpoint1 = torch.load(args.resume[0])
        checkpoint2 = torch.load(args.resume[1])
        Encoder.load_state_dict(checkpoint1['state_dict'])
        Decoder.load_state_dict(checkpoint2['state_dict'])
        print("=> loaded checkpoint '{}'".format(args.resume))
    else:
        print("=> !!!!!!! checkpoint found at '{}'".format(args.resume))
    for epoch in range(args.start_epoch, args.maxepoch):
        print("------------epoch-----------", epoch)
        adjust_learning_rate(optimizer_g, epoch)  # 调节器
        train_avg = train(Hnet=Encoder, Rnet=Decoder, optimizer=optimizer_g,
                          dataParser=trainDataLoader, epoch=epoch)
        val_avg = val(Hnet=Encoder, Rnet=Decoder, dataParser=valDataLoader, epoch=epoch)
        test(Hnet=Encoder, Rnet=Decoder, dataParser=testDataLoader, epoch=epoch)  # test直接用val就行
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
        output_name0 = 'Encode' + output_name_file_name % (epoch)
        output_name1 = 'Decode' + output_name_file_name % (epoch)
        if is_bestpth(val_avg['loss_avg']):
            # 对测试指标最高的进行测试
            save_model_name_stage0 = os.path.join(args.model_save_dir, 'Encode' + output_name0)
            save_model_name_stage1 = os.path.join(args.model_save_dir, 'Decode' + output_name1)
            torch.save(Encoder, save_model_name_stage0)
            torch.save(Decoder, save_model_name_stage1)
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
        secret = input_data['secret'].cuda()  # 没有进行归一化的secret，仅仅作为gt使用
        with torch.set_grad_enabled(True):
            images.requires_grad = True
            secret.require_grad = True
            optimizer.zero_grad()
            # encode
            contains = Hnet({"img": images, "msg": secret}).cuda()
            # 计算loss和noise layer无关，在noise layer之前
            loss_fn = lpips.LPIPS(net='alex').cuda()
            loss_lpips = loss_fn(images, contains)  # 输入和输出的cover进行判定  使用lpips函数，判定图像的相似度,越接近越小
            loss_lpips_mean = torch.mean(loss_lpips)  # 防止输出高维向量
            loss_l2 = nn.MSELoss()
            loss_res = loss_l2(images, contains)  # contain和contain_out的结果要接近，实际上和原本没有区别
            ssim_loss = SSIM()
            ssim = ssim_loss(images, contains)
            # 对于水印嵌入网络的loss计算已完成，可以进行contain的任意操作
            contains = noise_layer(contains)  # Hp只在此处使用，奇奇怪怪的参数看不懂，直接复制
            # decode net
            if epoch > 20:
                use_stn = True
            else:
                use_stn = False
            secret_out, stn_img = Rnet(contains, use_stn)  # 返回secret和矫正过的img
            print(secret_out.shape)
            loss_o_sec = cross_entropy_loss(secret_out, secret)  # loss2 输出的sec和输入的sec
            ent_losss = entropy_loss(secret_out)
            Loss_all = loss_o_sec + loss_lpips_mean * 0.5 + loss_res * 0.1 + ent_losss * 0.1 + (
                    1 - ssim) * 0.1  # 最终的loss
            # OUT secret下降很快，说明提取水印相对容易
            print("loss_o_sec----------", loss_o_sec)
            print(loss_lpips_mean)
            print(loss_res)
            print(ent_losss)
            writer.add_scalars('loss_gather', {'all': Loss_all.item(),
                                               'lpips': loss_lpips_mean.item(),
                                               'loss_l2': loss_res.item(),
                                               'ent_losss': ent_losss,
                                               'ssim_loss': 1 - ssim,
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
    Hnet.train()
    Rnet.train()
    end = time.time()
    for batch_index, input_data in enumerate(dataParser):
        # 读取数据的时间
        data_time.update(time.time() - end)
        images = input_data['image'].cuda()
        secret = input_data['secret'].cuda()  # 没有进行归一化的secret，仅仅作为gt使用
        with torch.set_grad_enabled(True):
            images.requires_grad = False
            secret.require_grad = False
            # encode
            contains = Hnet({"img": images, "msg": secret}).cuda()
            # 计算loss和noise layer无关，在noise layer之前
            loss_fn = lpips.LPIPS(net='alex').cuda()
            loss_lpips = loss_fn(images, contains)  # 输入和输出的cover进行判定  使用lpips函数，判定图像的相似度,越接近越小
            loss_lpips_mean = torch.mean(loss_lpips)  # 防止输出高维向量
            loss_l2 = nn.MSELoss()
            loss_res = loss_l2(images, contains)  # contain和contain_out的结果要接近，实际上和原本没有区别
            ssim_loss = SSIM()
            ssim = ssim_loss(images, contains)
            # 对于水印嵌入网络的loss计算已完成，可以进行contain的任意操作
            # contains = transform_net(contains, Hp, epoch)  # Hp只在此处使用，奇奇怪怪的参数看不懂，直接复制
            # decode net
            if epoch > 20:
                use_stn = True
            else:
                use_stn = False
            secret_out, stn_img = Rnet(contains, use_stn)  # 返回secret和矫正过的img
            loss_o_sec = cross_entropy_loss(secret_out, secret)  # loss2 输出的sec和输入的sec
            ent_losss = entropy_loss(secret_out)
            Loss_all = loss_o_sec + loss_lpips_mean * 0.5 + loss_res * 0.1 + ent_losss * 0.1 + (
                    1 - ssim) * 0.1  # 最终的loss
            print("loss_o_sec----------", loss_o_sec)
            print(loss_lpips_mean)
            print(loss_res)
            print(ent_losss)
            writer.add_scalars('loss_gather', {'all': Loss_all.item(),
                                               'lpips': loss_lpips_mean.item(),
                                               'loss_l2': loss_res.item(),
                                               'ent_losss': ent_losss,
                                               'ssim_loss': 1 - ssim,
                                               'loss_out_secret': loss_o_sec.item(),
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
def test(Hnet, Rnet, dataParser, epoch):  # 测试集
    Hnet.eval()
    Rnet.eval()
    for batch_index, input_data in enumerate(dataParser):
        if batch_index > 6:
            break
        images = input_data['image'].cuda()
        secret = input_data['secret'].cuda()
        with torch.set_grad_enabled(False):
            images.requires_grad = False
            contains = Hnet({"img": images, "msg": secret}).cuda()
            if epoch > 20:
                use_stn = True
            else:
                use_stn = False
            secret_out, stn_img = Rnet(contains, use_stn)  # 返回secret和矫正过的img

            writer.add_image('images:%d' % (batch_index),
                             make_grid(images, nrow=2), global_step=epoch)
            writer.add_image('container:%d' % (batch_index),
                             make_grid(contains, nrow=2), global_step=epoch)
            writer.add_text('out_secrect:%d' % (batch_index),
                            str(secret_out[0]), global_step=epoch)
            writer.add_text('in_secrect:%d' % (batch_index),
                            str(secret[0]), global_step=epoch)

def t():
    secret = np.random.binomial(1, 0.5, 100)
    secret = torch.from_numpy(secret).int()
    secret = torch.unsqueeze(secret, 0)
    print(secret[0].shape)

    writer.add_text('out_secrect:%d' % (1),
                    str(secret[0]), global_step=1)

if __name__ == '__main__':
    main()
