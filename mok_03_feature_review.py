#! /usr/local/bin/python3
# -*- coding:utf-8 -*-

"""
File Name: mok_03_feature_review.py 
Author: mok
Creation Date: 2020/6/11
Description: 
"""
from sklearn.feature_extraction import DictVectorizer  # 特征提取_字典矢量化器
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # 文本矢量化器
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 归一化和标准化
from sklearn.feature_selection import VarianceThreshold  # 特征选择_方差阈值
from sklearn.decomposition import PCA  # 特征降维_主成分分析
import jieba  # 中文分词
import numpy as np
from sklearn.impute import SimpleImputer  # 特征处理_插补


def dict_vector():
    """
    字典数据提取特征值
    :return:
    """
    # 初始化一个字典矢量化器
    dict_test = DictVectorizer(sparse=False)  # sparse指定是否在转换后生成矩阵

    # 输入数据并适应fit_rangsform()提取特征
    # 传入的是列表，列表中嵌套字典数据
    data = dict_test.fit_transform([{'city': '北京', 'temperature': 100},
                                    {'city': '上海', 'temperature': 80},
                                    {'city': '深圳', 'temperature': 50}])
    print(data)
    print(dict_test.get_feature_names())  # get_feature_names()拿到各个特征的名称
    print(dict_test.inverse_transform(data))  # inverse_transform()返回特征提取前的数据形式，这里与原本的有变化


def count_vector():
    """
    文本数据提取特征值
    """
    # 初始化文本矢量化器
    text_test = CountVectorizer()
    # 返回的是词频
    data = text_test.fit_transform(['life is short, i like python very much', 'i love python ahhhahaha'])

    print(data)
    print(text_test.get_feature_names())
    print(data.toarray())  # 将特征值转换为数组


def count_vector2():
    """
    中文文本数据提取特征值
    """
    text_cv = CountVectorizer()
    data = text_cv.fit_transform(['锄禾 日 当午', '床前明 月光', '春眠 不 觉 晓'])

    print(data)
    print(text_cv.get_feature_names())
    print(data.toarray())


def word_split():
    """
    中文词汇分割
    """
    text1 = jieba.cut('今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。')
    text2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")
    text3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")

    content1 = list(text1)
    content2 = list(text2)
    content3 = list(text3)

    str1 = ' '.join(content1)
    str2 = ' '.join(content2)
    str3 = ' '.join(content3)

    print(str1)
    print(str2)
    print(str3)

    return str1, str2, str3


def zh_count_vector3():
    """
    使用jieba对中文文本数据提取特征值
    """
    text1, text2, text3 = word_split()

    cv = CountVectorizer()
    text_data = cv.fit_transform([text1, text2, text3])

    print(text_data)
    print(text_data.toarray())


def tf_idf_vector():
    """
    倒排索引
    """

    text1, text2, text3 = word_split()

    tf_idf = TfidfVectorizer()
    # 返回的内容理解为，该词汇的特殊程度
    text_data = tf_idf.fit_transform([text1, text2, text3])

    print(text_data)
    print(tf_idf.get_feature_names())
    print(text_data.toarray())


def range_scaler():
    """特征值归一化处理"""
    # 初始化归一化器
    mms = MinMaxScaler(feature_range=(1, 2))  # feature_range指定归一的范围
    # 返回的是归一化后的特征值
    data = mms.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])

    print(data)


def standarize():
    """特征值标准化处理"""
    # 初始化标准化器
    sds = StandardScaler()
    # 返回特征值均值为0，标准差为1
    data = sds.fit_transform([[-1., 1., 3.2], [3., 4., 2.], [2., 6., -1.5]])

    print(data)


def deal_missing_value():
    """特征值缺失处理"""
    si = SimpleImputer(missing_values=np.nan, strategy='mean')  # missing_values指定缺失值是什么， strategy指定处理策略
    data = si.fit_transform([[1, 2], [np.nan, 3], [7, 6]])

    print(data)


def drop_low_variance():
    """特征值选择，删除低方差的特征"""
    # 初始化方差阈值过滤器
    vt = VarianceThreshold(threshold=1)  # threshold指定最小的方差
    data = vt.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])

    print(data)


def dimension_reduction():
    """特征值降维，主成分分析"""
    # 初始化PCA降维器
    pca = PCA(n_components=0.9)
    # 返回降维后的特征值
    data = pca.fit_transform([[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]])

    print(data)


def main():
    # count_vector()
    # count_vector2()
    # word_split()
    # zh_count_vector3()
    # tf_idf_vector()
    # range_scaler()
    # standarize()
    # deal_missing_value()
    # drop_low_variance()
    dimension_reduction()


if __name__ == '__main__':
    main()
