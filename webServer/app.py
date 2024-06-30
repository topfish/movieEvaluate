from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators


# 初始Flask实例,其作用是可以让Flask知道，在同一目录下可以找到HTML的模板文件夹templates
app = Flask(__name__)


# 继承From类，
class movieForm(Form):
    # 通过TextAreaField创建getDes表单
    # validators.DataRequired指定字段是必须的
    getdes = TextAreaField('', [validators.DataRequired()])


# 路由修饰器【‘/’】--->【index()】
@app.route('/')
def index():
    form = movieForm(request.form)
    # index()返回render_template渲染的start.html
    # 传入参数是movieForm实例化的表单
    return render_template('start.html', form=form)


def movie_eva(text):
    from joblib import load
    from sklearn.feature_extraction.text import HashingVectorizer
    from movieClassifier.classifier import preprocessor
    vect = HashingVectorizer(decode_error='ignore',
                             n_features=2 ** 21,
                             preprocessor=None,
                             tokenizer=preprocessor)
    # vect.transform参数需要一个list
    x = vect.transform([text])
    desmap = {1:"好电影", 0: "垃圾电影"}
    modelPath = '../movieClassifier/movie_SGDclassifier.model'
    model = load(modelPath)
    return desmap[model.predict(x)[0]]

# 路由修饰器【“/evaluate”】--->【evaluate()】,只接受POST方法
@app.route('/evaluate', methods=['POST'])
def evaluate():
    form = movieForm(request.form)
    # 如果方法是POST且表单验证通过
    if request.method == 'POST' and form.validate():
        # 获取输入的描述传递给模型，接收模型返回的结果
        des = movie_eva(request.form['getdes'])
        # 将获取到的des作为参数传递给eveluate.html一起渲染
        return render_template('evaluate.html', des=des)
    # 没有通过的情况，重新渲染start.html
    return render_template('start.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)
