#! /usr/local/bin/python3
# -*- coding:utf-8 -*-

"""
File Name: mok_05_fblocation.py
Author: mok
Creation Date: 2020/6/11
Description:
"""
from sklearn.datasets import load_iris, fetch_20newsgroups, load_boston  # 数据加载
from sklearn.model_selection import train_test_split, GridSearchCV  # 模型选择方法
from sklearn.neighbors import KNeighborsClassifier  # k邻近
from sklearn.preprocessing import StandardScaler  # 预处理标准化
from sklearn.feature_extraction.text import TfidfVectorizer  # 特征提取
from sklearn.naive_bayes import MultinomialNB  # 朴素贝叶斯
from sklearn.metrics import classification_report  # 混淆矩阵？
from sklearn.feature_extraction import DictVectorizer  # 特征提取
from sklearn.tree import DecisionTreeClassifier, export_graphviz  # 决策树
from sklearn.ensemble import RandomForestClassifier  # 随机森林
import pandas as pd


def knn():
    """k邻近算法"""
    # 获取数据
    datasets = pd.read_csv("./data/FBlocation/train.csv")
    # print(datasets.head())

    # 处理数据
    # 1.减少数据量
    data = datasets.query('x > 1.0 & x < 1.23 & y > 2.5 & y < 2.75')

    # 2.修改时间数据表示
    time_value = pd.to_datetime(data['time'], unit='s')
    # print(process_time)

    # 2.增加小时，天，星期几字段
    time_value = pd.DatetimeIndex(time_value)
    data.insert(data.shape[1], 'day', time_value.day)
    data.insert(data.shape[1], 'hour', time_value.hour)
    data.insert(data.shape[1], 'week', time_value.week)

    # 3.删除时间戳特征
    data = data.drop(['time'], axis=1)

    # 4.删除签到数低于3的目标位置
    # 得到位置参数总计
    place_count = data.groupby('place_id').count()
    tf = place_count[place_count.row_id > 3].reset_index()

    data = data[data['place_id'].isin(tf.place_id)]

    # 5.取出数据中特征值和目标值
    y = data['place_id']
    x = data.drop(['place_id'], axis=1)  # 去除place_id

    # 6.数据集分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 7.特征工程
    # std = StandardScaler()
    # x_train = std.fit_transform(x_train)
    # x_test = std.transform(x_test)

    # k邻近算法
    # 1.初始化k邻近算法分类器，指定邻居数n_neighbors
    knn = KNeighborsClassifier(n_neighbors=13)

    # 2.模型构建
    knn.fit(x_train, y_train)

    # 3.模型测试
    y_predict = knn.predict(x_test)
    print(y_predict)
    model_score = knn.score(x_test, y_test)
    print(model_score)

    # 进行网格搜索
    param = {"n_neighbors": [3, 5, 10, 12, 13]}
    gscv = GridSearchCV(knn, param_grid=param, cv=2)
    gscv.fit(x_train, y_train)

    # 预测准确率
    print("准确率：", gscv.score(x_test, y_test))
    print("交叉验证最好的结果：", gscv.best_score_)
    print("最好的模型：", gscv.best_estimator_)
    print("超参数每次交叉验证的结果：", gscv.cv_results_)


def main():
    knn()


if __name__ == '__main__':
    main()
