#! /usr/local/bin/python3
# -*- coding:utf-8 -*-

"""
File Name: mok_13_k_means.py
Author: mok
Creation Date: 2020/6/13
Description:
"""
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def k_means_classifier():
    """使用k-means对商品进行自动分类"""

    # 1.获取数据
    order_product = pd.read_csv("./instacart/order_products__prior.csv")
    products = pd.read_csv("./instacart/products.csv")
    orders = pd.read_csv("./instacart/orders.csv")
    aisles = pd.read_csv("./instacart/aisles.csv")

    # 2.数据处理
    # 合并表格
    table1 = pd.merge(order_product, products, on=["product_id", "product_id"])
    table2 = pd.merge(table1, orders, on=["order_id", "order_id"])
    table = pd.merge(table2, aisles, on=["aisle_id", "aisle_id"])

    # 合并交叉表
    table = pd.crosstab(table["user_id"], table["aisle"])

    # 数据截取
    train_table = table[:1000]
    test_table = table[1000:1200]

    # 3.特征工程
    pca = PCA(n_components=0.9)
    train_data = pca.fit_transform(train_table)
    # test_data = pca.transform(test_table)

    # 4.无监督学习
    kms = KMeans(n_clusters=9, random_state=24)
    cluster_labels = kms.fit_predict(train_data)

    # 5.模型评估
    silhouette_avg = silhouette_score(train_data, cluster_labels)
    print("聚类数:", 8,
          "轮廓系数:", silhouette_avg)


def main():
    k_means_classifier()


if __name__ == '__main__':
    main()
