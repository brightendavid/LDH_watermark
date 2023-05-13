# encoding: utf-8
# 参考
import functools
import torch
import torch.nn as nn
class Enet(nn.Module):
    def __init__(self, msg_size=100):
        super().__init__()
        self.msg_dense = nn.Sequential(
            nn.Linear(msg_size, 7500),
            nn.ReLU(inplace=True))
        self.up_x8 = nn.Upsample(scale_factor=8)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(inplace=True))
        self.up6 = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(256, 128, kernel_size=(2, 2)),
            nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True))  # 7
        self.up7 = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(128, 64, kernel_size=(2, 2)),
            nn.ReLU(inplace=True))  # 8
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True))  # 9
        self.up8 = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(64, 32, kernel_size=(2, 2)),
            nn.ReLU(inplace=True))  # 10
        self.conv8 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True))  # 11
        self.up9 = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(32, 32, kernel_size=(2, 2)),
            nn.ReLU(inplace=True))  # 12
        self.conv9 = nn.Sequential(
            nn.Conv2d(70, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True))  # 13
        self.residual = nn.Conv2d(32, 3, kernel_size=(1, 1))  # 13

        self.up_x2 = nn.Upsample(scale_factor=2)
        self.up_x8 = nn.Upsample(scale_factor=8)
    def forward(self, image, msg):
        batch = image.data.size()[0]
        msg = msg - .5
        image = image - .5

        msg = self.msg_dense(msg)  # msg 100 => 7500   先进行dense,再进行reshape
        msg = msg.view(batch, 3, 50, 50)  # msg 50*50*3
        msg_enlarged = self.up_x8(msg)  # msg_enlarged 400*400*3
        inputs = torch.cat((msg_enlarged, image), dim=1)  # inputs 400*400*6

        conv1 = self.conv1(inputs)  # conv1 400*400*32
        conv2 = self.conv2(conv1)  # conv2 200*200*32
        conv3 = self.conv3(conv2)  # conv3 100*100*64
        conv4 = self.conv4(conv3)  # conv4 50*50*128
        conv5 = self.conv5(conv4)  # conv5 25*25*256
        up6 = self.up6(self.up_x2(conv5))  # up6 50*50*256->50*50*128
        merge6 = torch.cat((conv4, up6), dim=1)  # merge6 50*50*256
        conv6 = self.conv6(merge6)  # conv6 50*50*128
        up7 = self.up7(self.up_x2(conv6))  # up7 100*100*128->100*100*64
        merge7 = torch.cat((conv3, up7), dim=1)  # merge7 100*100*128
        conv7 = self.conv7(merge7)  # conv7 100*100*64
        up8 = self.up8(self.up_x2(conv7))  # up8 200*200*64->200*200*32
        merge8 = torch.cat((conv2, up8), dim=1)  # merge8 200*200*64
        conv8 = self.conv8(merge8)  # conv8 200*200*32
        up9 = self.up9(self.up_x2(conv8))  # up9 400*400*32->400*400*32
        merge9 = torch.cat((conv1, up9, inputs), dim=1)  # merge9 400*400*(32+32+4)
        conv9 = self.conv9(merge9)  # conv9 400*400*32
        residual = self.residual(conv9)  # residual 400*400*3
        return residual


if __name__ == '__main__':
    # 测试运行
    model = Enet()
    a = torch.zeros((1, 3, 400, 400))
    msg = torch.zeros((1, 100))
    b = model(a,msg)
    print("out shape", b.shape)
    # total_params, total_trainable_params = cnn_paras_count(model)
    # print(total_params*4/1024/1024,"mb")
