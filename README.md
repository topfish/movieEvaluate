# movieEvaluate
电影评分


# 项目结构
## movieClassifier  
模型训练
### 这里用了两种方法训练模型：
#### 逻辑二分法
代码文件：classifier.py
这里使用了网格搜索+pipeline的方法进行参数调优

生成的模型保存为：movie_classifier.model
#### 随机梯度下降法
代码文件：SGDclassifier.py
这里是和外学习法，批量训练数据，模型可以通过新增数据持续学习。

生成的模型保存为：movie_SGDclassifier.model