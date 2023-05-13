#!/usr/bin/env python
# -*- coding:utf-8 -*-
import logging
import gradio as gr
from torchvision import transforms
from test.test_watermark2 import gen_data
import numpy as np
import cv2 as cv
from Functions.loss_functions import *
from model_ldh import Revealnet, Hide_net, Encode_net

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)
torch.cuda.empty_cache()
is_cuda = True
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if is_cuda:
    Rnet = Revealnet.RevealNet(input_nc=3,
                               output_nc=1, norm_layer=nn.BatchNorm2d,
                               output_function=nn.Sigmoid).cuda()
    Enet = Encode_net.UnetGenerator(input_nc=1,
                                    output_nc=1, num_downs=5, norm_layer=nn.BatchNorm2d,
                                    output_function=nn.Sigmoid).cuda()
    Hnet = Hide_net.HideNet(input_nc=2,
                            output_nc=3, norm_layer=nn.BatchNorm2d,
                            output_function=nn.Tanh).cuda()
else:
    Rnet = Revealnet.RevealNet(input_nc=3,
                               output_nc=1, norm_layer=nn.BatchNorm2d,
                               output_function=nn.Sigmoid)
    Enet = Encode_net.UnetGenerator(input_nc=1,
                                    output_nc=1, num_downs=5, norm_layer=nn.BatchNorm2d,
                                    output_function=nn.Sigmoid)
    Hnet = Hide_net.HideNet(input_nc=2,
                            output_nc=3, norm_layer=nn.BatchNorm2d,
                            output_function=nn.Tanh)

model_path0 = r"../save_model/0710/BEST_Encode_07-10watermark_checkpoint49dict.pth"
model_path1 = model_path0.replace("Encode", "Hide")
model_path2 = model_path0.replace("Encode", "Rev")

checkpoint0 = torch.load(model_path0, map_location=device)
checkpoint1 = torch.load(model_path1, map_location=device)
checkpoint2 = torch.load(model_path2, map_location=device)
Hnet.load_state_dict(checkpoint1)
Enet.load_state_dict(checkpoint0)
Rnet.load_state_dict(checkpoint2)
Enet.eval()
Hnet.eval()
Rnet.eval()




def tensor2np(src):
    output = src.squeeze(0)
    output = np.array(output.cpu().detach().numpy(), dtype='float32')
    output = np.transpose(output, (1, 2, 0))
    output = np.where(output > 1, 1, output)
    return output


def vc_fn(input_cover, input_message):
    secret = gen_data(word=input_message)
    # secret = secret[:h, :w]
    data = transforms.Compose([
        transforms.ToTensor()
    ])(input_cover)  # 张量化
    data = data[np.newaxis, :, :, :]
    data = data.cuda()
    secret = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])(secret)  # 张量化
    secret = secret[np.newaxis, :, :, :]
    secret = secret.cuda()
    encode_S = Enet(secret).cuda()  # 生成的残差 输入的是secrets
    res = Hnet(data, encode_S)  # 3通道
    res1 = tensor2np(res)
    cv.imwrite("test_picture/res.png", res1 * 255 * 15)
    out_contain = res + data
    print(input_message)
    out_image = tensor2np(out_contain)
    print(out_image)
    print(out_image.shape)
    return out_image


def vc_reveal(input_cover):
    print(input_cover.shape)
    input_cover = input_cover[::2, ::2]
    src = transforms.Compose([
        transforms.ToTensor()
    ])(input_cover)  # 张量化
    src = src[np.newaxis, :, :, :]
    if is_cuda:
        src = src.type(torch.cuda.FloatTensor)
    else:
        src = src.type(torch.FloatTensor)
    out_sec = Rnet(src)
    print(out_sec.shape)
    out_image = tensor2np(out_sec)
    print(out_image.shape)
    out_image = out_image[:, :, 0]
    print(out_image.shape)
    return out_image


app = gr.Blocks()
with app:
    with gr.Tabs():
        with gr.TabItem("Hide watermark"):
            gr.Markdown(value="""webui""")
            with gr.Row():
                vc_input1_h = gr.Image(label="上传cover图像(1024,2048)")
                vc_input2_h = gr.Textbox(label="Watermark Message", default="F1h3AlKw5\n445.342.796.448\n939073726462")
            vc_submit_h = gr.Button("嵌入", variant="primary")
            with gr.Column():
                vc_output_h = gr.Image(label="Output container")
        vc_submit_h.click(vc_fn, [vc_input1_h, vc_input2_h], [vc_output_h])
        with gr.TabItem("Reveal"):
            gr.Markdown(value="""webui""")
            with gr.Row():
                vc_input1_r = gr.Image(label="上传container图像(不要太大)")
                vc_output_r = gr.Image(label="Output secret")
            vc_submit_r = gr.Button("提取", variant="primary")

        vc_submit_r.click(vc_reveal, [vc_input1_r], [vc_output_r])
    app.launch(share=True,show_error=True)
