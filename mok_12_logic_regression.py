#! /usr/local/bin/python3
# -*- coding:utf-8 -*-

"""
File Name: mok_12_logic_regression.py 
Author: mok
Creation Date: 2020/6/13
Description: 
"""

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report
import joblib
import pandas as pd
import numpy as np


def logistic_regression_cancer():
    """使用逻辑回归进行癌症预测"""
    # 1.获取数据
    # 构造列标签名字
    column = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
              'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
              'Mitoses', 'Class']
    # 加载数据
    cancer_datasets = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
        names=column)

    # 2.数据处理
    # 将？数据处理成缺失值
    cancer_datasets = cancer_datasets.replace(to_replace='?', value=np.nan)
    # 去除缺失值
    cancer_datasets = cancer_datasets.dropna()

    # 3.数据分割
    x_train, x_test, y_train, y_test = train_test_split(cancer_datasets[column[1:10]],
                                                        cancer_datasets[column[10]], test_size=0.25)

    # 4.特征工程
    # 标准化
    # 初始标准化器
    sds = StandardScaler()
    x_train = sds.fit_transform(x_train)
    x_test = sds.transform(x_test)

    # 5.逻辑回归预测
    # 初始化逻辑回归器
    lg = LogisticRegression()

    # 开始训练
    lg.fit(x_train, y_train)

    # 查看预测值
    y_predict = lg.predict(x_test)
    # print(y_predict)

    # 查看准确率与召回率
    print('准确率：', lg.score(x_test, y_test))
    print('混淆矩阵：', classification_report(y_test, y_predict, labels=[2, 4], target_names=['良性', '恶性']))


def main():
    logistic_regression_cancer()


if __name__ == '__main__':
    main()
