#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : conf.py
@Time    : 2020/3/17 16:23
@desc	 : 项目的配置文件
'''

# 定义采用数据集的路径
data_path = '../../docs/dataset/dataset.csv'

# 定义训练集数据的维度，即样本的特征数
dimen = 9

# 定义数据标准化模型的路径
scaler_path = '../../docs/scaler/scaler.pkl'

# 定义朴素贝叶斯模型的保存路径
nb_model_path = '../../docs/model/nb_clf.pkl'

# 定义支持向量机模型的保存路径
svm_model_path = '../../docs/model/svm_clf.pkl'

# 定义朴神经网络模型的保存路径
nn_model_path = '../../docs/model/nn_clf.pkl'
