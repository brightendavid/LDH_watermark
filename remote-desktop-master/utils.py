import threading
import time

from test.test_watermark2 import Secret_message, Hide_secret_Enet_only

S = Secret_message()
secret = S.secret
global enet_feat
global DesktopSrc
DesktopSrc = "1"
enet_feat = Hide_secret_Enet_only(secret)


class EncodeThread(threading.Thread):  # 继承父类threading.Thread
    def __init__(self, sleep_time=5):
        # 设置睡眠时间为5秒，每隔5秒执行一次
        threading.Thread.__init__(self)
        self.sleep_time = sleep_time

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        while 1:
            global enet_feat
            enet_feat = Hide_secret_Enet_only(secret)
            time.sleep(self.sleep_time)


class HideThread(threading.Thread):  # 继承父类threading.Thread
    def __init__(self, data, enet_feat):
        threading.Thread.__init__(self)
        self.data = data
        self.enet_feat = enet_feat

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        print("H")
        # global enet_feat
        # torch.cuda.empty_cache()
        # global DesktopSrc
        # DesktopSrc = Hide_Hnet_only(self.data, self.enet_feat)


def Hide_desktop(data):
    """
    这个函数没有测试过，当心了，开始进程
    @param data: 桌面图像
    @return: 加入水印的桌面图像
    """
    thread1 = EncodeThread()
    thread1.setDaemon(True)
    thread1.start()

    thread2 = HideThread(data, enet_feat=enet_feat)
    thread2.setDaemon(True)
    thread2.start()

    # while 1:
    #     global enet_feat
    #     torch.cuda.empty_cache()
    #     global DesktopSrc
    #     DesktopSrc = Hide_Hnet_only(data, enet_feat)


if __name__ == "__main__":
    Hide_desktop("1")
