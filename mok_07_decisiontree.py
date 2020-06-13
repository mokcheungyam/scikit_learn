#! /usr/local/bin/python3
# -*- coding:utf-8 -*-

"""
File Name: mok_07_.py 
Author: mok
Creation Date: 2020/6/13
Description: 
"""

from sklearn.datasets import load_iris, fetch_20newsgroups, load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def decision_tree():
    """
    决策树对泰坦尼克号人员进行生死预测
    """

    # 1.数据获取
    titan_datasets = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

    # 2.数据预处理
    # 选择特征值和目标值
    x = titan_datasets[['pclass', 'age', 'sex']]
    y = titan_datasets['survived']

    # 删除缺失值，并将原数据处填充平均值
    x['age'].fillna(x['age'].mean(), inplace=True)

    # 3.数据分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 4.特征工程
    # 初始化字典矢量化器
    dict_vector = DictVectorizer(sparse=False)

    # 特征提取
    x_train = dict_vector.fit_transform(x_train.to_dict(orient='records'))

    # print(dict_vector.get_feature_names())
    # print(x_train)
    # print(type(x_train))

    # 5.决策树预测
    # 初始化决策树估计器
    decide_tree = DecisionTreeClassifier(max_depth=8)

    # 使用训练集进行训练
    decide_tree.fit(x_train, y_train)

    # 导出决策树结构
    export_graphviz(decide_tree, out_file='./tree.dot',
                    feature_names=['年龄', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', '女性', '男性'])


def main():
    decision_tree()


if __name__ == '__main__':
    main()
