#! /usr/local/bin/python3
# -*- coding:utf-8 -*-

"""
File Name: mok_04_algorithm_review.py 
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


def get_basic_datasets():
    """数据的获取及其属性查看"""
    # 1.鸢尾花数据集
    # 直接从sklearn的datasets中加载鸢尾花的数据集
    # iris_datasets = load_iris()

    # print(iris_datasets)  # 返回一个字典，可以根据key来获取字典中的内容
    # print("-" * 100)
    # print(iris_datasets['target'])  # 目标值标签
    # print("-" * 100)
    # print(iris_datasets.DESCR)  # 数据集描述

    # 数据分割
    # 返回训练集和测试集，test_size指定测试集的大小比例
    # x_train, x_test, y_train, y_test = train_test_split(iris_datasets['data'],
    #                                                     iris_datasets['target'],
    #                                                     test_size=0.25)
    # print(x_train)
    # print("-" * 100)
    # print(y_train)
    # print("-" * 100)
    # print(x_test)
    # print("-" * 100)
    # print(y_test)
    # print("-" * 100)

    # 2.新闻分类数据集
    # news_datasets = fetch_20newsgroups(subset='all')
    # print(news_datasets)
    # print(news_datasets.DESCR)
    # print(news_datasets.target)

    # 3.波士顿房价数据集
    # house_price_datasets = load_boston()
    #
    # print(house_price_datasets.data)
    # print("-" * 100)
    # print(house_price_datasets.target)
    # print("-" * 100)
    # print(house_price_datasets.DESCR)


def main():
    get_basic_datasets()


if __name__ == '__main__':
    main()
