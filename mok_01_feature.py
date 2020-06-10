from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import jieba
import numpy as np
from sklearn.impute import SimpleImputer

from sklearn.feature_extraction.text import CountVectorizer


def dict_vec():
    """
    字典数据提取
    """
    dict_test = DictVectorizer(sparse=False)
    data = dict_test.fit_transform([{'city': '北京', 'temperature': 100},
                                    {'city': '上海', 'temperature': 60},
                                    {'city': '深圳', 'temperature': 30}])
    print(data)
    print(dict_test.get_feature_names())
    print(dict_test.inverse_transform(data))


def count_vec():
    """
    对文本进行特征值化，计数提取
    """
    vector = CountVectorizer()
    res = vector.fit_transform(["life is is short,i like python", "life is too long,i dislike python"])

    print(vector.get_feature_names())
    print(res)
    print(res.toarray())


def count_vec2():
    """
    对中文文本进行特征值化，单个汉字单个字母不统计
    """
    cv = CountVectorizer()
    data = cv.fit_transform(["人生 苦短，我 喜欢 python", "人生漫长，不用 python"])

    print(cv.get_feature_names())
    print(data)
    print(data.toarray())


def cut_word():
    """
    使用jieba进行词语分割
    """
    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")
    con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")
    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")

    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)

    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)

    return c1, c2, c3


def zh_hans_vec():
    """
    对中文文本进行特征值化
    """
    c1, c2, c3 = cut_word()
    print(c1, c2, c3)

    cv = CountVectorizer()
    data = cv.fit_transform([c1, c2, c3])

    print(cv.get_feature_names())
    print(data.toarray())


def tf_idf_vec():
    """
    对中文文本进行特征值化，倒排索引
    """
    c1, c2, c3 = cut_word()
    print(c1, c2, c3)

    tf = TfidfVectorizer()
    data = tf.fit_transform([c1, c2, c3])

    print(tf.get_feature_names())
    print(data.toarray())


def mm():
    """
    归一化处理
    """
    mm = MinMaxScaler(feature_range=(1, 3))
    data = mm.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])

    print(data)


def standardize():
    """
    标准化处理
    """
    std = StandardScaler()
    data = std.fit_transform([[1., -1., 3.], [2., 4., 2.], [4., 6., -1.]])

    print(data)


def im():
    """
    缺失值处理，缺失值形式为NaN，nan，对于?先使用replace替换
    """
    im = SimpleImputer(missing_values=np.nan, strategy='mean')
    data = im.fit_transform([[1, 2], [np.nan, 3], [7, 6]])

    print(data)


def var():
    """
    特征选择-删除低方差的特征
    """
    var = VarianceThreshold(threshold=1)
    data = var.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])

    print(data)


def pca():
    """
    主成分分析进行特征降维
    """
    pca = PCA(n_components=0.9)
    pca2 = PCA(n_components=1)

    data = pca.fit_transform([[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]])
    data2 = pca2.fit_transform([[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]])

    print(data)
    print(data2)


def main():
    # dict_vec()
    # count_vec()
    # count_vec2()
    # zh_hans_vec()
    # tf_idf_vec()
    # mm()
    # standardize()
    # im()
    # var()
    pca()


if __name__ == "__main__":
    main()
