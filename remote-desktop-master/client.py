#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
远程桌面代码，做的功能还行
可以使用，在配置中给出对应的端口号和Ip地址
目前看来，只有传输桌面影像的功能，可能由控制功能，在同一台机器上做远程控制有困难，鼠标无法移动到窗口内部，就被传送到桌面的边缘；从这点看，可能具有控制功能
可以参考桌面的抓取和传输
再添加桌面水印添加，即可实现本项目功能
基于socket和网络编程实现
看不懂

client 接收端
"""
from PIL import ImageGrab
from pynput.mouse import Controller, Button
from configparser import RawConfigParser  # 这个类使得ini配置文件生效
import numpy as np
import socket
import sys
import cv2
import time
import struct
import json
import hashlib
import win32api
import win32con
import threading

# pyinstaller 打包时需添加 --hidden-import=pynput.keyboard._win32 --hidden-import=pynput.mouse._win32
from test.test_watermark2 import Hide_Hnet_only, Hide_secret_Enet_only, Secret_message

resolution = (win32api.GetSystemMetrics(win32con.SM_CXSCREEN), win32api.GetSystemMetrics(win32con.SM_CYSCREEN))
# resolution 获取桌面的分辨率大小
resize = (960, 544)  # 新生成的远程桌面窗口，按照当前桌面分辨率进行resize


class MyConfigParser(RawConfigParser):
    # 配置解析器?  这个类是什么意思
    def __init__(self, defaults=None):
        RawConfigParser.__init__(self, defaults=defaults)

    def optionxform(self, option_str):
        return option_str


def socket_client(host, port):
    # socket服务器端
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        print(s.recv(1024).decode())  # recv函数 接收数据
    except socket.error as e:
        print(e)
        sys.exit(1)

    resize_ratio = (resolution[0] / resize[0], resolution[1] / resize[1])  # resize比例

    base_info = {
        'resize_ratio': resize_ratio
    }
    s.send(json.dumps(base_info).encode())  # python对象转换成json对象
    while True:
        response = s.recv(1024)
        if response.decode() == "client info confirm":
            break

    receive_thread = threading.Thread(target=receive_mouse_msg, args=(s,))  # 线程，接收鼠标事件
    receive_thread.start()
    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 95]
    # todo Enet 线程，在while的前面，修改特征
    thread1 = EncodeThread()
    thread1.setDaemon(True)
    thread1.start()

    while True:
        flag, msg = make_screen_img(encode_param)
        if not flag:
            break
        flag = send_msg(s, msg)
        if not flag:
            break
        time.sleep(0.01)
    s.close()


def make_screen_img(encode_param):
    # 抓取桌面并改为jpg编码方式
    try:
        screen = ImageGrab.grab()  # 抓取桌面
        bgr_img = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)  # 颜色空间转换, cv2.COLOR_RGB2BGR 将RGB格式转换成BGR格式
        img = cv2.resize(bgr_img, resize)  # 缩放图片
        # todo 加入添加水印的函数 加入提前计算secret就行了，提前计算就是每分钟调整一次secret，cover实时更新 .这一段感觉古怪
        watermark = True
        if watermark:
            global enet_feat
            DesktopSrc = Hide_Hnet_only(img, enet_feat)
            img = DesktopSrc

        return True, cv2.imencode(".png", img, encode_param)[1].tostring()  # 把当前图片img按照jpg格式编码 可以修改
    except Exception as e:
        print(e)
        return False, None


global enet_feat
S = Secret_message()
secret = S.secret
enet_feat = Hide_secret_Enet_only(secret)


class EncodeThread(threading.Thread):  # 继承父类threading.Thread
    def __init__(self, sleep_time=5):
        # 设置睡眠时间为5秒，每隔5秒执行一次
        threading.Thread.__init__(self)
        self.sleep_time = sleep_time

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        while 1:
            try:
                global enet_feat
                enet_feat = Hide_secret_Enet_only(secret)
                time.sleep(self.sleep_time)
            except Exception as e:
                print(e)


def get_msg_info(msg):
    return len(msg), hashlib.md5(msg).hexdigest()


def make_msg_header(msg_length, msg_md5):
    header = {
        'msg_length': msg_length,
        'msg_md5': msg_md5
    }
    return json.dumps(header).encode()


def send_msg(conn, msg):
    msg_length, msg_md5 = get_msg_info(msg)
    msg_header = make_msg_header(msg_length, msg_md5)
    msg_header_length = struct.pack('i', len(msg_header))
    try:
        header_len_res = conn.send(msg_header_length)
        header_res = conn.send(msg_header)
        msg_res = conn.sendall(msg)
        return True
    except socket.error as e:
        print(e)
        return False


def receive_mouse_msg(conn, ):
    mouse = Controller()  # server的鼠标移动到
    while True:
        try:
            msg_length = struct.unpack('i', conn.recv(4))[0]
            mouse_msg = json.loads(conn.recv(msg_length).decode())
            mouse_position = mouse_msg.get('mouse_position')
            event = mouse_msg.get('event')
            flags = mouse_msg.get('flags')
            mouse_event(mouse, mouse_position[0], mouse_position[1], event, flags)
            # print(mouse_position[0], mouse_position[1], event, flags)

        except Exception as e:
            print(e)
            break
    conn.close()


def mouse_event(mouse, x, y, event, flags):
    # 鼠标事件响应
    flag_event = get_flag_event(flags)
    mouse.position = (x, y)
    # 鼠标左键
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse.press(Button.left)
    elif event == cv2.EVENT_LBUTTONUP:
        mouse.release(Button.left)
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        mouse.click(Button.left, 2)
    # 鼠标中键
    elif event == cv2.EVENT_MBUTTONDOWN:
        mouse.press(Button.middle)
    elif event == cv2.EVENT_MBUTTONUP:
        mouse.release(Button.middle)
    elif event == cv2.EVENT_MBUTTONDBLCLK:
        mouse.click(Button.middle, 2)
    # 鼠标右键
    elif event == cv2.EVENT_RBUTTONDOWN:
        mouse.press(Button.right)
    elif event == cv2.EVENT_RBUTTONUP:
        mouse.release(Button.right)
    elif event == cv2.EVENT_RBUTTONDBLCLK:
        mouse.click(Button.right, 2)


def get_flag_event(value):
    flags = [
        cv2.EVENT_FLAG_LBUTTON,  # 1
        cv2.EVENT_FLAG_RBUTTON,  # 2
        cv2.EVENT_FLAG_MBUTTON,  # 4
        cv2.EVENT_FLAG_CTRLKEY,  # 8
        cv2.EVENT_FLAG_SHIFTKEY,  # 16
        cv2.EVENT_FLAG_ALTKEY,  # 32
    ]
    flag_events = []
    for flag in sorted(flags, reverse=True):
        if value >= flag:
            flag_events.append(flag)
            value -= flag
    return flag_events


if __name__ == '__main__':
    config = MyConfigParser()
    config.read('config.ini', encoding='utf-8')  # 读取ip和端口  server
    server_host = config.get('Server', 'host')  # 获取ip
    server_port = config.getint('Server', 'port')  # 获取端口号
    socket_client(server_host, server_port)  # 启动client服务器端  接收端，获取发送过来的桌面
