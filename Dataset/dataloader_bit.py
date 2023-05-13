"""
dataloader 读取原本数据集的src部分作为cover或者secrect
考虑到只需要将二值图像作为秘密信息加入,secrect可以考虑只有二值图，单通道
secrect图像使用生成的文本图，读取的数据集是用于cover图像
一般使用数据集DIV2K_valid_HR
dataloade 基本没有问题
没有对cover进行归一化，归一化之后，生成的contain就花了
"""
import os
import traceback
import cv2 as cv
import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from Dataset.secret_data import gen_bit_data
from glob import glob


def change_root_by_device(device=r"../data/DIV2K_train_HR"):
    """
    返回数据集路径
    """
    if device == 'sjw':
        # sjw 笔记本 移动硬盘路径
        data_root = r"F:\dataset_watermark\DIV2K_valid_HR"
    elif device == 'test':
        data_root = 'F:\dataset_watermark\DIV2K_train_HR'
    else:
        data_root = r"../data/DIV2K_train_HR"

    return data_root


class Bitdataset:
    def __init__(self, device="test", val_percent=0.1, train_val_test_mode='train'):
        self.secret_size = 100
        self.pic_size = (400, 400)
        self.root = change_root_by_device(device)
        self.image_list = glob(os.path.join(self.root, '*.png'))  # 返回list
        self.train_val_test_mode = train_val_test_mode
        self.train_list, self.val_list = \
            train_test_split(self.image_list, test_size=val_percent,
                             train_size=1 - val_percent, random_state=1000)

    def __getitem__(self, index):
        if self.train_val_test_mode == "train":
            path = self.train_list[index]
        elif self.train_val_test_mode == "val":
            path = self.val_list[index]
        else:
            # test
            path = self.val_list[index]

        try:
            img = Image.open(path).convert('RGB')
            img = ImageOps.fit(img, self.pic_size)  # 大图就crop,小图就周围补齐

            secrect = gen_bit_data(self.secret_size)

            # 水印类代码一般不对cover加入归一化,包括DDH和UDH，因为DDH总还是把原图弄回来
            img_in = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.4484666, 0.43753183, 0.40452665), (0.28013384, 0.2657674, 0.28883007)),
            ])(img)

            return {'image': img_in, 'secret': secrect}
        # img_in 是经过了归一化的，img_out是没有经过归一化的
        except Exception as e:
            traceback.print_exc(e)

    def __len__(self):
        # 这里重要，不是原本的总list,而是分割后的list
        if self.train_val_test_mode == "train":
            return len(self.train_list)
        elif self.train_val_test_mode == "val":
            return len(self.val_list)



def shou_tensor_img(tensor_img: torch.Tensor):
    """
    显示tensor
    """
    to_pil = torchvision.transforms.ToPILImage()
    img = tensor_img.cpu().clone()
    img = to_pil(img)
    img = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
    cv.imshow("img", img)
    cv.waitKey(0)


if __name__ == '__main__':
    testdataset = Bitdataset(device="sjw")
    dataloader = torch.utils.data.DataLoader(dataset=testdataset, batch_size=3, num_workers=1)
    print(iter(dataloader))
    image_input, secret_input = next(iter(dataloader))  # 返回的是迭代器的第一个，iter()生成了一个迭代器
    print(type(image_input), type(secret_input))
    print(image_input.shape, secret_input.shape)
    print(image_input.max())
    # shou_tensor_img(image_input[0])

