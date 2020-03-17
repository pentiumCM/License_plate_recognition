#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : neural_network_clf.py
@Time    : 2020/3/17 21:58
@desc	 : 神经网络分类器 - 多层感知器 MLP
'''

import numpy as np
import pandas as pd
from sklearn import preprocessing
import joblib as jl
import operator
import os
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

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


def nn_classifier(train_data, train_labels, model_file):
    """
    采用神经网络分类器，进行分类

    :param train_data: 训练集数据
    :param train_labels: 训练集标签
    :param model_file: 保存的模型文件路径
    :return: 模型文件保存到本地
    """

    clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                        beta_1=0.9, beta_2=0.999, early_stopping=False,
                        epsilon=1e-08, hidden_layer_sizes=(5, 2),
                        learning_rate='constant', learning_rate_init=0.001,
                        max_iter=200, momentum=0.9, n_iter_no_change=10,
                        nesterovs_momentum=True, power_t=0.5, random_state=1,
                        shuffle=True, solver='lbfgs', tol=0.0001,
                        validation_fraction=0.1, verbose=False, warm_start=False)
    clf.fit(train_data, train_labels)

    # save model
    jl.dump(clf, model_file)

    print("神经网络模型保存成功")


if __name__ == '__main__':
    # 定义采用数据集的路径
    data_path = conf.data_path

    # 定义训练集数据的维度，即样本的特征数
    dimen = conf.dimen

    # 定义数据标准化模型的路径
    scaler_path = conf.scaler_path

    # 定义神经模型的保存路径
    model_path = conf.nn_model_path

    # 构造数据集，并且保存数据标准化模型，方便标准化新样本
    data, labels = make_data_set(data_path, dimen, scaler_path)

    # 划分数据集为：训练集和测试集
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.4, random_state=0)

    # 构建分类器
    nn_classifier(data_train, labels_train, model_path)

    # load model
    clf2 = jl.load(model_path)
    # y_pred = clf2.predict(data)
    score = clf2.score(data_test, labels_test)
    print("准确率：", score)

    # print("高斯朴素贝叶斯，样本总数： %d 错误样本数 : %d" % (data.shape[0], (labels != y_pred).sum()))
