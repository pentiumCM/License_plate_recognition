#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : bilinear_interpolation.py
@Time    : 2020/3/7 11:01
@desc	 : 双线性插值,来调整图像的尺寸
'''

import numpy as np
import cv2

import math


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
            dst_img[i,
            j,
            :] = (1. - src_y_float) * (1. - src_x_float) * src_img[src_y_int,
                                                           src_x_int,
                                                           :] + (1. - src_y_float) * src_x_float * src_img[src_y_int,
                                                                                                   src_x_int + 1,
                                                                                                   :] + src_y_float * (
                             1. - src_x_float) * src_img[src_y_int + 1,
                                                 src_x_int,
                                                 :] + src_y_float * src_x_float * src_img[src_y_int + 1,
                                                                                  src_x_int + 1,
                                                                                  :]
    return dst_img


if __name__ == '__main__':
    img_path = 'F:/experiment/image_identification/License_plate_recognition/Img/1.jpg'

    src = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # 高337 - 宽500
    src_shape = (src.shape[0], src.shape[1])
    # 定义图片缩放后的尺寸
    dst_shape = (100, 400)

    # 图像放缩均采用双线性插值法
    # opencv的放缩图像函数
    resize_image = cv2.resize(src, (400, 100), interpolation=cv2.INTER_LINEAR)

    # 自定义的图像放缩函数
    dst_img = bilinear(src, dst_shape)

    cv2.imwrite('F:/experiment/image_identification/License_plate_recognition/Img/1_new_resize.jpg', resize_image)
    cv2.imwrite('F:/experiment/image_identification/License_plate_recognition/Img/1_new.jpg', dst_img)
