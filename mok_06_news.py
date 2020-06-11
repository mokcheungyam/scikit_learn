#! /usr/local/bin/python3
# -*- coding:utf-8 -*-

"""
File Name: mok_06_news.py 
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


def navie_bayes():
    """朴素贝叶斯文本分类"""
    # 获取数据
    datasets = fetch_20newsgroups(subset='all')

    # 处理数据
    # 1.数据分割
    x_train, x_test, y_train, y_test = train_test_split(datasets.data, datasets.target, test_size=0.25)
    # 2.特征提取
    tf = TfidfVectorizer()
    x_train = tf.fit_transform(x_train)
    x_test = tf.transform(x_test)

    # 朴素贝叶斯算法
    mltnb = MultinomialNB(alpha=1.0)
    mltnb.fit(x_train, y_train)
    y_predict = mltnb.predict(x_test)
    print(y_predict)
    print("-" * 100)
    model_score = mltnb.score(x_test, y_test)
    print(model_score)


def main():
    navie_bayes()


if __name__ == '__main__':
    main()
