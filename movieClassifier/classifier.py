import os
import tarfile
import requests
from tqdm import tqdm
import logging
import pandas as pd
import pyprind
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from joblib import dump


# 日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_file(url, filename):
    # 发送GET请求
    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # 确保请求成功

        # 获取文件总大小
        total_size = int(r.headers.get('content-length', 0))

        # 创建一个进度条
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

        # 打开文件准备写入
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # 过滤掉保持连接的chunk
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        # 关闭进度条
        progress_bar.close()


def data_check():

    source = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    target = 'aclImdb_v1.tar.gz'

    # 数据集不存在则下载
    if not os.path.isdir('aclImdb') and not os.path.isfile('aclImdb_v1.tar.gz'):
        logger.info("下载数据集")
        download_file(source, target)

    # 解压缩
    if not os.path.isdir('aclImdb'):
        logger.info("开始解压缩")
        with tarfile.open(target, 'r:gz') as tar:
            tar.extractall()


def data_to_csv():
    if os.path.isfile("movie_data.csv"):
        logger.info("have movie_data.csv")
        return
    logger.info("data to csv")
    basepath = 'aclImdb'

    labels = {'pos': 1, 'neg': 0}
    pbar = pyprind.ProgBar(50000)  # 进度条
    data_list = []
    for s in ('test', 'train'):
        for l in ('pos', 'neg'):
            path = os.path.join(basepath, s, l)
            for file in sorted(os.listdir(path)):
                with open(os.path.join(path, file),
                          'r', encoding='utf-8') as infile:
                    txt = infile.read()
                data_list.append([txt, labels[l]])
                pbar.update()  # 进度条更新
    pbar.stop()  # 关闭进度条
    df = pd.DataFrame(data_list)
    df.columns = ['review', 'sentiment']

    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))

    df.to_csv('movie_data.csv', index=False, encoding='utf-8')

    # df = pd.read_csv('movie_data.csv', encoding='utf-8')


# 通过正则匹配处理文本里的html和表情等字符
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text


def text_split(text):
    return text.split()


def text_porter(text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]


def main():
    # 检查原始数据集是否存在
    data_check()

    # 原始数据处理，标记评价文本和标签，转存为dataForm
    data_to_csv()

    # 读取csv数据
    df = pd.read_csv('movie_data.csv', encoding='utf-8')
    # 处理html和表情等字符
    df['review'] = df['review'].apply(preprocessor)
    # 词干提取
    # todo 下载停词表
    # nltk.download('stopwords')
    stop = stopwords.words('english')
    # 训练集和测试集
    X_train = df.loc[:25000, 'review'].values
    y_train = df.loc[:25000, 'sentiment'].values
    X_test = df.loc[25000:, 'review'].values
    y_test = df.loc[25000:, 'sentiment'].values
    # 词频
    tfidf = TfidfVectorizer(strip_accents=None,
                            lowercase=False,
                            preprocessor=None)
    # 参数优化取值
    param_grid = [{'vect__ngram_range': [(1, 1)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [text_split, text_porter],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]},
                  {'vect__ngram_range': [(1, 1)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [text_split, text_porter],
                   'vect__use_idf': [False],
                   'vect__norm': [None],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]},
                  ]
    # pipeline
    lr_tfidf = Pipeline([('vect', tfidf),
                         ('clf', LogisticRegression(random_state=0, solver='liblinear'))])

    # 通过网格搜索法确认最佳参数组合
    gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                               scoring='accuracy',
                               cv=5,
                               verbose=2,
                               n_jobs=-1)
    # 开始训练模型
    gs_lr_tfidf.fit(X_train, y_train)

    print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
    print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
    # 取最佳的参数作为模型
    clf = gs_lr_tfidf.best_estimator_
    # 测试集验证
    print('Test Accuracy: %.3f' % clf.score(X_test, y_test))

    # 保存模型
    logger.info("dump model to movie_classifier.model")
    dump(clf, 'movie_classifier.model')


if __name__ == "__main__":
    main()
