def writeData(file):
    print("Loading raw data...")
    raw_data = pd.read_csv(file, header=None, low_memory=False)
    # return raw_data
import numpy as np
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
from sklearn import preprocessing
raw_data_filename = "MachineLearning1/MachineLearningCSV/expendData/totall_extend.csv"
print("Loading raw data...")
raw_data = pd.read_csv(raw_data_filename, header=None, low_memory=False)
raw_data = raw_data.sample(frac=0.05)
last_column_index = raw_data.shape[1] - 1
print("print data labels:")
print(raw_data[last_column_index].value_counts())
# print("Transforming data...")
raw_data[last_column_index], attacks = pd.factorize(raw_data[last_column_index], sort=True)
features = raw_data.iloc[:, :raw_data.shape[1] - 1]  # pandas中的iloc切片是完全基于位置的索引
labels = raw_data.iloc[:, raw_data.shape[1] - 1:]
features = preprocessing.scale(features)
features = pd.DataFrame(features)
labels = labels.values.ravel()
df = pd.DataFrame(features)
X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=0.8, test_size=0.2, stratify=labels)
# print("X_train,y_train:", X_train.shape, y_train.shape)
# print("X_test,y_test:", X_test.shape, y_test.shape)
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
print("Training model...")
clf = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=2, min_samples_split=2,splitter="best")
trained_model = clf.fit(X_train, y_train)
print("Score:", trained_model.score(X_train, y_train))#训练集的分数
# 预测
print("Predicting...")
y_pred = clf.predict(X_test)
print("Computing performance metrics...")
results = confusion_matrix(y_test, y_pred)
error = zero_one_loss(y_test, y_pred)
print(error)
# 根据混淆矩阵求预测精度
list_diag = np.diag(results)
list_raw_sum = np.sum(results, axis=1)
print("Predict accuracy of the decisionTree: ", np.mean(list_diag) / np.mean(list_raw_sum))
# # 测试模型：
# print("Testing model...")
# clf = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=2,min_samples_split=2, splitter="best")
# tested_model = clf.fit(X_test, y_test)
# print("Score:", trained_model.score(X_test, y_test))
# 预测
print("Training...")
y_pred = clf.predict(X_train)
print("Computing performance metrics...")
results = confusion_matrix(y_train, y_pred)
error = zero_one_loss(y_train, y_pred)  # 评估模型，评估出错误率
print(error)
# 根据混淆矩阵求预测精度
list_diag = np.diag(results)
list_raw_sum = np.sum(results, axis=1)
print("Predict accuracy of the decisionTree: ", np.mean(list_diag) / np.mean(list_raw_sum))


#浅尝网络搜索：
#第一步确定好那个基尼还是熵
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
#比较决策树划分标准对模型影响：
# DT = DecisionTreeClassifier(random_state = 25)
# score1 = cross_val_score(DT,X_train,y_train,cv=10).mean()
# print(f'基尼指数得分：{score1}')
# DT = DecisionTreeClassifier(criterion = 'entropy',random_state = 25)
# score2 = cross_val_score(DT,X_train,y_train,cv=10).mean()
# print(f'熵得分：{score2}')
#第二得到最优深度：
import matplotlib.pyplot as plt
test = []
for i in range(20):
    clf = DecisionTreeClassifier(max_depth=i + 1

                                 , criterion="entropy"

                                 , random_state=30

                                 , splitter="random"

                                 )
    clf = clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    test.append(score)
plt.plot(range(1, 21), test, color="red", label="max_depth")
plt.legend()
plt.show()

#第三单独看看min_samples_split的变化趋势
# ScoreAll = []
# for i in range(2,25):
#     DT = DecisionTreeClassifier(max_depth = 10,min_samples_split = i,random_state = 30)#最优深度是19来着的
#     score = cross_val_score(DT,X_train,y_train,cv=10).mean()
#     ScoreAll.append([i,score])
# ScoreAll = np.array(ScoreAll)
#
# max_score = np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0] ##这句话看似很长的，其实就是找出最高得分对应的索引
# print("最优参数以及最高得分:",ScoreAll[max_score])
# # print(ScoreAll[,0])
# plt.figure(figsize=[15,7])
#
# plt.plot(ScoreAll[:,0],ScoreAll[:,1])
# plt.show()

# ###第四调min_samples_leaf这个参数
# from sklearn.model_selection import cross_val_score
# import matplotlib.pyplot as plt
# import numpy as np
# ScoreAll = []
# for i in range(1,30):
#     DT = DecisionTreeClassifier(min_samples_leaf = i,max_depth = 19,random_state = 25)#min_samples_split =2)
#     score = cross_val_score(DT,X_train,y_train,cv=10).mean()
#     ScoreAll.append([i,score])
# ScoreAll = np.array(ScoreAll)
#
# max_score = np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0] ##这句话看似很长的，其实就是找出最高得分对应的索引
# print("最优参数以及最高得分:",ScoreAll[max_score])
# # print(ScoreAll[,0])
# plt.figure(figsize=[18,7])
# plt.plot(ScoreAll[:,0],ScoreAll[:,1])
# plt.show()



















