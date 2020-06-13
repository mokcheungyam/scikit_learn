#! /usr/local/bin/python3
# -*- coding:utf-8 -*-

"""
File Name: mok_08_random_forest.py 
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
    随机森林对泰坦尼克号人员进行生死预测
    """

    # 1.数据获取
    titan_datasets = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

    # 2.数据预处理
    # 选择特征值和目标值
    x = titan_datasets.loc[:, ('pclass', 'age', 'sex')]
    y = titan_datasets.loc[:, 'survived']

    # 删除缺失值，并将原数据处填充平均值
    x['age'].fillna(x['age'].mean(), inplace=True)

    # 3.数据分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 4.特征工程
    # 初始化字典矢量化器
    dict_vector = DictVectorizer(sparse=False)

    # 特征提取
    x_train = dict_vector.fit_transform(x_train.to_dict(orient='records'))
    x_test = dict_vector.transform(x_test.to_dict(orient="records"))

    # 5.随机森林预测
    # 初始化随机森林估计器
    random_forest = RandomForestClassifier(n_jobs=-1)

    # 自定义n_estimators与max_depth
    rf_param = {'n_estimators': [120, 200, 300, 500, 800, 1100], 'max_depth': [5, 8, 15, 25, 30]}

    # 网格搜索与交叉验证
    gscv = GridSearchCV(random_forest, param_grid=rf_param, cv=2)

    # 开始训练
    gscv.fit(x_train, y_train)

    # 输出结果
    print('预测的准确率：', gscv.score(x_test, y_test))
    print('最佳参数模型', gscv.best_params_)


def main():
    decision_tree()


if __name__ == '__main__':
    main()
