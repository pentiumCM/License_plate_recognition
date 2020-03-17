#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : svm_clf.py
@Time    : 2020/2/19 22:12
@desc	 : 车牌定位的SVM分类器
'''

import os
import numpy as np
import pandas as pd
import joblib as jl
import operator
from functools import reduce
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import license_plate_localization.conf as conf


def make_data_set(data_file, dimens, scaler_path):
    """
    读取数据集中的数据，制作数据集

    :param data_file: 数据集路径
    :param dimens: 训练集的维度
    :param scaler_path: 数据标准化模型的路径
    :return: 返回训练集，data,labels
    """
    # 读取数据集CSV文件，并且过滤掉表头
    # dataset = pd.read_csv(data_file, skiprows=1)
    dataset = pd.read_csv(data_file)
    # 获取样本数据集中的特征数据
    # features = ['横坐标X', '纵坐标Y', '宽度', '高度', '面积']
    data = dataset.iloc[0:, 0:dimens]  # 从第一行到最后一行，第1列到第dimens列
    # 获取样本的类别标签labels
    labels = dataset.iloc[0:, dimens:]  # 从第一行到最后一行，第dimens+1列

    # 调整数据集数组的维数
    data = np.reshape(np.array(data), (-1, dimens))
    labels = np.reshape(np.array(labels), (1, -1))
    labels = reduce(operator.add, labels)  # 将标签二维数组转化为一维数组

    # 数据标准化
    # 如果数据标准化模型文件不存在，重新创建
    if not os.path.exists(scaler_path):
        # 数据标准化
        scaler = preprocessing.StandardScaler().fit(data)  # 用训练集数据训练scaler
        X_scaled = scaler.transform(data)  # 用其转换训练集是理所当然的
        # 保存数据标准化的模型
        jl.dump(scaler, scaler_path)
        print("数据标准化模型保存成功")
    else:
        # 加载数据标准化的模型
        scaler = jl.load(conf.scaler_path)
        # 用标准化模型来标准化数据
        X_scaled = scaler.transform(data)

    return X_scaled, labels


def svm_classifier(train_data, train_labels, model_file):
    """
    采用SVM分类器，进行分类

    :param train_data: 训练集数据
    :param train_labels: 训练集标签
    :param model_file: 保存的模型文件路径
    :return: 模型文件保存到本地
    """

    # X = [[0, 0], [1, 1]]
    # y = [0, 1]

    # 主要调节的参数有：C、kernel、degree、gamma、coef0。
    clf = svm.SVC(C=0.4, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
                  max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)
    clf.fit(train_data, train_labels)

    # save model
    jl.dump(clf, model_file)

    print("模型文件保存成功")


if __name__ == '__main__':
    # 定义采用数据集的路径
    data_path = conf.data_path

    # 定义训练集数据的维度
    dimen = conf.dimen

    # 定义数据标准化模型的路径
    scaler_path = conf.scaler_path

    # 定义生成模型的保存路径
    model_path = conf.svm_model_path

    # 构造数据集
    data, labels = make_data_set(data_path, dimen, scaler_path)

    # 划分数据集为：训练集和测试集
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.4, random_state=0)

    # 构建分类器
    svm_classifier(data_train, labels_train, model_path)

    # load model
    clf2 = jl.load(model_path)
    # y_pred = clf2.predict(data)
    score = clf2.score(data_test, labels_test)
    print("准确率：", score)

    # print("SVM分类器，样本总数： %d 错误样本数 : %d" % (data.shape[0], (labels != y_pred).sum()))
