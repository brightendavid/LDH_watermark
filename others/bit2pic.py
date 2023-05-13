import string
import numpy as np
import cv2 as cv
def readbit(path):
    with open(path, "r") as f:
        lines = f.read()  # 读取全部内容
        list = []  ## 空列表, 将第i行数据存入list中
        for word in lines.split():
            word = word.strip(string.whitespace)
            if word != '[' and word != ']':
                list.append(word)
        print(len(list))
        return list
def bitTomask(bit_list):
    src = np.zeros((1000,1000))
    for i in range(len(bit_list)):
        if bit_list[i] == "True":
            a = i//10
            b =i%10
            src[a*10:a*11,b*10:b*11] = 0
        else:
            a = i // 10
            b = i % 10
            src[a * 100:a * 110, b * 100:b * 110] = 255
    return src
if __name__ == '__main__':
    path = r"C:\Users\brighten\Desktop\new\extract\0801"
    list = readbit(path)
    src = bitTomask(list)
    cv.imshow("a",src)
    cv.waitKey(0)