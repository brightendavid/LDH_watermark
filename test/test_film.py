# 处理录屏数据
import cv2
from test.test_watermark import Reveal_one_pic


def film_detect(film_path):
    # VideoCapture方法是cv2库提供的读取视频方法
    cap = cv2.VideoCapture(film_path)
    # 设置需要保存视频的格式“xvid”
    # 该参数是MPEG-4编码类型，文件名后缀为.avi
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 设置视频帧频
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 设置视频大小
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # VideoWriter方法是cv2库提供的保存视频方法
    # 按照设置的格式来out输出
    out = cv2.VideoWriter('C:\\Users\\brighten\\Desktop\\out.avi',fourcc ,fps, size)

    # 确定视频打开并循环读取
    while(cap.isOpened()):
        # 逐帧读取，ret返回布尔值
        # 参数ret为True 或者False,代表有没有读取到图片
        # frame表示截取到一帧的图片
        ret, frame = cap.read()
        if ret == True:
            # 垂直翻转矩阵
            # frame = cv2.flip(frame,0)
            # frame =frame[500:1000,1200:1800]
            # cv2.imshow('frame', frame)
            # cv2.waitKey(100)
            # print(type(frame))
            frame = Reveal_one_pic(frame,is_cuda=False)
            print(type(frame))
            frame =cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
            out.write(frame)
            # print(frame.shape)
            cv2.imshow('frame',frame)
            cv2.waitKey(100)
            cv2.imwrite("test_picture/11.png", frame)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break
    # 释放资源
    cap.release()
    out.release()
    # 关闭窗口
    cv2.destroyAllWindows()


if __name__ == '__main__':
    film_detect(r"C:\Users\brighten\Desktop\软件服务外包\测试视频1.mp4")
