import pandas as pd
from movieClassifier.classifier import preprocessor
from joblib import load
from movieClassifier.classifier import text_split

def load_model(text):
    # model的输入和输出皆为list
    # 所以接收的数据集的格式为: ["string"]
    # 返回的结果集的格式为：["int"]
    model = load('./movieClassifier/movie_classifier.model')
    return model.predict([text])[0]


def load_SGDmodel(text):
    # model的输入使用HashingVectorizer向量化后的数据
    from movieClassifier.classifier import preprocessor
    from sklearn.feature_extraction.text import HashingVectorizer
    vect = HashingVectorizer(decode_error='ignore',
                             n_features=2 ** 21,
                             preprocessor=None,
                             tokenizer=preprocessor)
    # vect.transform参数需要一个list
    x = vect.transform([text])
    model = load('./movieClassifier/movie_SGDclassifier.model')
    return model.predict(x)



def main():

    # 模型加载验证
    text1 = "i love this movie, this movie is soo good and nice"
    text2 = "this movie is so bad bad bad"
    text3 = "this move is just so so"
    texts = [text1, text2, text3]
    for t in texts:
        res = load_model(t)
        # res = load_SGDmodel(t)
        if res:
            print("this is good movie")
        else:
            print("this is bad movie")


if __name__ == "__main__":
    main()