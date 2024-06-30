# movieClassifier

通过电影评价内容的情感分析，对电影进行分类

本项目使用了情感分析的方法，情感分析不只能用于电影评价的分类，也可以用于舆情的监控和分析，比如监控媒体/论坛上的留言内容是否有涉及舆情的言论等。


# 项目结构
## movieClassifier  
模型训练
这里用了两种方法训练模型：

1：逻辑二分法

代码文件：classifier.py

这里使用了网格搜索+pipeline的方法进行参数调优

生成的模型保存为：movie_classifier.model

2：随机梯度下降法

代码文件：SGDclassifier.py

这里是和外学习法，批量训练数据，模型可以通过新增数据持续学习。

生成的模型保存为：movie_SGDclassifier.model

## webServer
web服务端，使用flask框架开发。

### 开始

运行webServer目录下的app.py即可

python3 app.py

## test.py
这个时验证文件，用于测试验证功能

在web首页：
1. 输入对电影的评价（目前只支持英文评价）
2. 点击【开始分析】按钮
3. 经模型分析后，显示此电影的分类