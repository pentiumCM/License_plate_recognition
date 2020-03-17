"""
function:检测与定位车牌
author:王池社
time:2020-1-27
"""
import cv2
import numpy as np

import math
import joblib as jl
import os

# 定义常量
minArea = 2000  # 车牌区域允许最大面积

# 加载分类器model
clf = jl.load('../../docs/model/svm_clf.pkl')
# 加载数据标准化的模型
scaler = jl.load('../../docs/scaler/scaler.pkl')

'''获取图片的像素点，再重新计算像素点输出图片'''
'''论文A Novel Least Squre and Image Rotation based Method for ...第5995页的描述实现'''


def getPiex(img):
    row = img.shape[0]  # 图片的行
    col = img.shape[1]  # 图片的列
    for i in range(row - 1):
        for j in range(col - 1):
            if (int(img[i + 1, j]) - int(img[i, j]) >= 50):
                img[i, j] = 255
            else:
                img[i, j] = 0

    return img


# 灰度拉伸函数
def stretch(img):
    max = float(img.max())
    min = float(img.min())

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = (255 / (max - min)) * img[i, j] - (255 * min) / (max - min)

    return img


def bilinear(src_img, dst_shape):
    """
    双线性插值法,来调整图片尺寸

    :param org_img: 原始图片
    :param dst_shape: 调整后的目标图片的尺寸
    :return:    返回调整尺寸后的图片矩阵信息
    """
    dst_img = np.zeros((dst_shape[0], dst_shape[1], 3), np.uint8)
    dst_h, dst_w = dst_shape
    src_h = src_img.shape[0]
    src_w = src_img.shape[1]
    # i：纵坐标y，j：横坐标x
    # 缩放因子，dw,dh
    scale_w = src_w / dst_w
    scale_h = src_h / dst_h

    for i in range(dst_h):
        for j in range(dst_w):
            src_x = float((j + 0.5) * scale_w - 0.5)
            src_y = float((i + 0.5) * scale_h - 0.5)

            src_x_int = math.floor(src_x)
            src_y_int = math.floor(src_y)

            src_x_float = src_x - src_x_int
            src_y_float = src_y - src_y_int

            if src_x_int + 1 == src_w or src_y_int + 1 == src_h:
                dst_img[i, j, :] = src_img[src_y_int, src_x_int, :]
                continue
            # print(src_x_int, src_y_int)
            dst_img[i, j, :] = (1. - src_y_float) * (1. - src_x_float) * src_img[src_y_int, src_x_int, :] + \
                               (1. - src_y_float) * src_x_float * src_img[src_y_int, src_x_int + 1, :] + \
                               src_y_float * (1. - src_x_float) * src_img[src_y_int + 1, src_x_int, :] + \
                               src_y_float * src_x_float * src_img[src_y_int + 1, src_x_int + 1, :]
    return dst_img


def lpr(filename):
    # 读取图片
    orgImg = cv2.imread(filename, cv2.IMREAD_COLOR)
    '''图像预处理'''
    # 压缩图片到指定大小
    image = cv2.resize(orgImg, (360, 580))

    # 缩放图片，替换 cv2.resize
    # src_shape = (orgImg.shape[0], orgImg.shape[1])
    # dst_shape = (math.ceil(src_shape[0] / 2), math.ceil(src_shape[1] / 2))

    # 自定义的图像放缩函数
    # image = bilinear(orgImg, dst_shape)

    # RGB转灰色
    grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 灰度拉伸(灰的更灰，白的更白）
    stretchedImg = stretch(grayImg)

    # 直方图均衡Histogram Equalization。全局图像质量的提升，可能会丢失一些局部图像的细节。比如，脸被画出来了，
    # 但是脸上的鼻子的轮廓可能就没有了，所以有时需要采用自适应直方图均衡
    HEImg = cv2.equalizeHist(stretchedImg)
    # cv2.imshow("HEImg", HEImg)
    # 自适应直方图均衡。局部图像均衡化
    # HEImgOrg = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # HEImg = HEImgOrg.apply(grayImg)
    # cv2.imshow("HEImg", HEImg)

    # 输出的ndarray列表的维度483行300列 print(grayImg.shape)
    # 高斯滤波平滑处理，第4个参数设为0，表示不计算y方向的梯度，原因是车牌上的数字在竖方向较长，重点在于得到竖方向的边界
    GaussianBlurImg = cv2.GaussianBlur(HEImg, (3, 3), 0)
    # 采用Sobel进行边缘提取
    SobelImg = cv2.Sobel(GaussianBlurImg, -1, 1, 0, ksize=3)
    # 图像二值化。采用像素间的计算
    # binaryImg = getPiex(SobelImg)

    # #使用Canny函数进行边缘检测
    # CannyImg = cv2.Canny(grayImg, grayImg.shape[0], grayImg.shape[1])

    # 通过最大类间方差（OTSU法）进行二值化
    ret, binaryImg = cv2.threshold(SobelImg, 127, 255, cv2.THRESH_OTSU)
    # ret, binaryImg = cv2.threshold(grayImg, 127, 255, cv2.THRESH_BINARY)
    # ret, binaryImg = cv2.threshold(SobelImg, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imshow("binaryImg", binaryImg)

    '''通过形态学方法，顶帽-闭运算-开运算-膨胀，消除小区域，保留大块区域，从而定位车牌'''
    # 形态学运算设置卷积核kernel
    kernel = np.ones((5, 15), np.uint8)
    # kernel = np.ones((5, 15), np.uint8)
    # 图像顶帽运算
    topHatImg = cv2.morphologyEx(binaryImg, cv2.MORPH_TOPHAT, kernel)
    # cv2.imshow("topHatImg", topHatImg)
    # 先进行闭运算
    closeImg = cv2.morphologyEx(topHatImg, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("closeImg", closeImg)
    # 再进行开运算
    openImg = cv2.morphologyEx(closeImg, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("openImg", openImg)
    # 由于部分图像得到的轮廓边缘不整齐，因此再进行一次膨胀操作
    # 调整iterations的值会将结果调整好
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilationImg = cv2.dilate(openImg, element, iterations=3)
    # cv2.imshow("dilationImg", dilationImg)

    '''通过区域进行车牌定位'''
    # 获取轮廓
    contours, hierarchy = cv2.findContours(dilationImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 过滤面积小的区域
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > minArea]

    '''画出这些框的最小外接矩形--20200212修改'''
    '''函数 cv2.minAreaRect() 返回一个Box2D结构rect：（最小外接矩形的中心（x，y），（宽度，高度），旋转角度）'''
    carContours = []
    for cnt in contours:
        # 循环这些边缘区域，取得这些区域的最小外接矩形
        rect = cv2.minAreaRect(cnt)
        # 排除矩形区域中不符合车牌长宽比的矩形
        areaWidth, areaHeight = rect[1]
        if areaWidth < areaHeight:
            areaWidth, areaHeight = areaHeight, areaWidth
        whRatio = areaWidth / areaHeight  # 计算长宽比
        # 要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
        if whRatio > 2 and whRatio < 5.5:

            carContours.append(rect)
            box = cv2.boxPoints(rect)
            print("rect：", rect)
            # print("cnt：",cnt)
            # print("cnt[1][0]：",cnt[1][0])

            # 规整为矩形
            a = []
            b = []
            for point in cnt:
                b.append(point[0][0])
                a.append(point[0][1])
            r = [min(b), min(a), max(b), max(a)]
            block = image[r[1]:r[3], r[0]:r[2]]

            # RGB转换为HSV
            hav = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)

            # 求均值
            mean = cv2.mean(hav)

            h = mean[0]
            s = mean[1]
            v = mean[2]

            # print('中国')
            # print("box：",box)
            box = np.int0(box)

            # 对新的样本进行预测
            area = areaWidth * areaHeight  # 求区域的面积
            x, y = rect[0]
            angle = rect[2]
            sample = np.array([x, y, areaWidth, areaHeight, angle, area, h, s, v])
            sample = np.reshape(sample, (1, -1))

            # 引用数据标准化模型
            global scaler
            sample_scaler = scaler.transform(sample)

            # 引用模型分类器
            global clf
            label = clf.predict(sample_scaler)
            print("类别标签：", label)
            # 如果预测的标签为0，则退出当前，执行下一循环
            if (label == 1):
                cv2.drawContours(image, [box], 0, (255, 0, 0), 2)
            else:
                cv2.drawContours(image, [box], 0, (0, 255, 255), 2)

    # 将轮廓规整为长方形
    # rectangles = []
    # for c in contours:
    #     x = []
    #     y = []
    #     for point in c:
    #         y.append(point[0][0])
    #         x.append(point[0][1])
    #     r = [min(y), min(x), max(y), max(x)]
    #     rectangles.append(r)

    # 用颜色识别出车牌区域
    # 需要注意的是这里设置颜色识别下限low时，可根据识别结果自行调整
    distR = []
    maxMean = 0
    # 在得到的矩形框中进行循环
    # 获得矩形框的区域的个数：
    # print(len(rectangles))
    maxweight, maxindex = 0, -1
    # for r in rectangles:
    #     block = image[r[1]:r[3], r[0]:r[2]]
    #     # RGB转HSV
    #     hav = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
    #     # 蓝色车牌范围
    #     low = np.array([100, 50, 50])
    #     up = np.array([140, 255, 255])
    #     # 根据阈值构建掩模
    #     # 低于low的高于up的图像素变为0，其余部分变为255
    #     result = cv2.inRange(hav, low, up)
    #     # 用计算均值的方式找蓝色最多的区域
    #     mean = cv2.mean(result)
    #     if mean[0] > maxMean:
    #         maxMean = mean[0]
    #         distR = r

    # cv2.rectangle(image, (distR[0] + 3, distR[1]), (distR[2] - 3, distR[3]), (0, 255, 0), 2)
    # cv2.rectangle(image, (distR[0], distR[1]), (distR[2], distR[3]), (0, 255, 0), 2)

    cv2.imshow("image" + str(i + 1) + ".jpg", image)
    # cv2.imshow("grayImg", grayImg)
    # cv2.imshow("stretchedImg", stretchedImg)
    # cv2.imshow("openingImg", openingImg)
    # cv2.imshow("strImg",strImg)
    # cv2.imshow("SobelImg",SobelImg)
    # cv2.imshow("binaryImg",binaryImg)
    # cv2.imshow("closingImg",closingImg)
    # cv2.imshow("openingImgSec",openingImgSec)
    # cv2.imshow("dilationImg", dilationImg)
    cv2.waitKey(0)


if __name__ == "__main__":

    # 循环读取文件
    path = "F:/experiment/image_identification/License_plate_recognition/Img/illumination/"
    i = 0
    path_list = os.listdir(path)
    # print(path_list)
    for filename in path_list:
        lpr(path + filename)
        i = i + 1
