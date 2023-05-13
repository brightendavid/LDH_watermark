import os

import cv2
import numpy as np


def file2pics(video_path=r'D:\ApowerREC\20220810_105015.mp4', output_path=r'D:/ApowerREC/true_desktop2/', interval=5):
    # 视频地址
    # 输出文件夹
    # 每间隔10帧取一张图片
    num = 1
    vid = cv2.VideoCapture(video_path)
    while vid.isOpened():
        is_read, frame = vid.read()
        if is_read:
            if num % interval == 1:
                file_name = '%08d' % num
                cv2.imwrite(output_path + str(file_name) + '.png', frame)
                # 00000111.jpg 代表第111帧
                cv2.waitKey(1)
            num += 1
        else:
            break


global frame_old
frame_old = 0


def picdiff(video_path=r'D:\ApowerREC\20220811_091600.mp4', output_path=r'D:/ApowerREC/diff_origin/', interval=20):
    num = 1
    vid = cv2.VideoCapture(video_path)

    while vid.isOpened():
        is_read, frame_new = vid.read()
        if is_read:
            if num % interval == 1:
                file_name = '%08d' % num
                try:
                    global frame_old
                    frame = np.abs(frame_old - frame_new)
                    cv2.imwrite(output_path + str(file_name) + '.png', frame * 200)
                    # 00000111.jpg 代表第111帧
                    cv2.waitKey(1)
                    frame_old = frame_new
                except Exception as e:
                    print(e)

            num += 1

        else:
            break


import cv2 as cv


def check_diff(path_dir=r"C:\Users\brighten\Desktop\pics\save_result_cover", out_path=r"C:\Users\brighten\Desktop\pics\cover"):
    srcs = []
    for index, item in enumerate(os.listdir(path_dir)):
        img_path = os.path.join(path_dir, item)
        src = cv.imread(img_path)
        srcs.append(src)
    global src_old
    src_old = srcs[0]
    for i in range(len(srcs)):
        diff = np.abs(src_old-srcs[i])
        item = str(i) + ".png"
        img_path = os.path.join(out_path, item)
        cv.imwrite(img_path, diff)
        src_old = srcs[i]


if __name__ == '__main__':
    check_diff()
