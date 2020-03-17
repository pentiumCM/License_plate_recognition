# !/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : *********@qq.com
@Software: PyCharm
@File    : linear_regression.py
@Time    : 2019/12/19 0:19
@desc	 : 线性回归示例
'''

import numpy as np
from sklearn import preprocessing
import joblib as jl

# 构建训练数据
X = np.array([[90, 50], [70, 40], [80, 60]])

# 数据标准化
# 用训练集数据训练scaler
scaler = preprocessing.StandardScaler().fit(X)

# 保存数据标准化的模型
jl.dump(scaler, '../../docs/scaler/scaler_1.pkl')

import joblib as jl

# 加载数据标准化的模型
scaler_1 = jl.load('../../docs/scaler/scaler_1.pkl')

sample = np.reshape(np.array([100, 60]), (1, -1))
# 标准化新样本
sample_scaler = scaler.transform(sample)

print(sample_scaler)
