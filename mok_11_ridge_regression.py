#! /usr/local/bin/python3
# -*- coding:utf-8 -*-

"""
File Name: mok_11_ridge_regression.py 
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


def ridge_regression_house_price():
    """使用岭回归预测Boston房价"""
    # 1.获取数据
    house_price_datasets = load_boston()

    # 2.数据分割
    x_train, x_test, y_train, y_test = train_test_split(house_price_datasets.data,
                                                        house_price_datasets.target,
                                                        test_size=0.25)

    # 3.特征工程
    # 初始化标准化器，特征值和目标值
    sds_x = StandardScaler()
    sds_y = StandardScaler()

    # 特征值标准化
    x_train = sds_x.fit_transform(x_train)
    x_test = sds_x.transform(x_test)

    # 目标值标准化
    y_train = sds_y.fit_transform(y_train.reshape(-1, 1))
    y_test = sds_y.transform(y_test.reshape(-1, 1))

    # 4.线性回归预测
    # 初始化线性回归估计器
    rr = Ridge(alpha=1.0)  # 岭回归下降

    # 开始训练
    rr.fit(x_train, y_train.ravel())  # 使用.ravel()解决警告

    # 查看回归系数
    # print(lr.coef_)

    # 使用模型进行预测
    y_predict = rr.predict(x_test)

    # 查看预测结果
    y_predict_restore = sds_y.inverse_transform(y_predict)
    # print(y_predict_restore)

    # 查看均方误差
    y_test_restore = sds_y.inverse_transform(y_test)
    print('岭回归下均方误差：', mean_squared_error(y_test_restore, y_predict_restore))

    # 5.保存模型
    joblib.dump(rr, "./tmp/rrtest3.pkl")


def main():
    ridge_regression_house_price()


if __name__ == '__main__':
    main()
