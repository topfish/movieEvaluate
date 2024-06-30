# 核外学习
# 相比于逻辑二分法的算法，这个明显速度快多了

from classifier import preprocessor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import pyprind
import numpy as np
from joblib import dump


# 按行读取‘movie_data.csv’的生成器
# 效果：每次读一个文档并返回‘review’和‘sentiment’
def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


# 读取size个评价，并将评价的‘review’和‘sentiment’返回
# 如果doc_stream中不足size个评价，则报错并返回空
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


def main():
    # 向量化
    vect = HashingVectorizer(decode_error='ignore',
                             n_features=2 ** 21,
                             preprocessor=None,
                             tokenizer=preprocessor)

    # 实例化随机梯度下降法
    clf = SGDClassifier(loss='log_loss', random_state=1)

    # 实例化一个读数据的迭代器
    doc_stream = stream_docs(path='movie_data.csv')

    # 创建一个进度条
    pbar = pyprind.ProgBar(45)

    classes = np.array([0, 1])
    # 分批训练（45批），每次用1000条数据，总计用45000条训练数据
    for _ in range(45):
        X_train, y_train = get_minibatch(doc_stream, size=1000)
        if not X_train:
            break
        X_train = vect.transform(X_train)
        clf.partial_fit(X_train, y_train, classes=classes)
        pbar.update()

    # 用5000条数据作为测试数据
    X_test, y_test = get_minibatch(doc_stream, size=5000)
    X_test = vect.transform(X_test)
    # 验证一下测试集
    print('Accuracy: %.3f' % clf.score(X_test, y_test))

    # 用测试数据更新模型
    clf = clf.partial_fit(X_test, y_test)

    # 报错模型
    dump(clf, 'movie_SGDclassifier.model')


if __name__ == "__main__":
    main()
