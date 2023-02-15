def writeData(file):
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
raw_data = raw_data.sample(frac=0.03)
last_column_index = raw_data.shape[1] - 1
print("print data labels:")
print(raw_data[last_column_index].value_counts())
raw_data[last_column_index], attacks = pd.factorize(raw_data[last_column_index], sort=True)
features = raw_data.iloc[:, :raw_data.shape[1] - 1]  # pandas中的iloc切片是完全基于位置的索引
labels = raw_data.iloc[:, raw_data.shape[1] - 1:]
features = preprocessing.scale(features)
features = pd.DataFrame(features)
labels = labels.values.ravel()
df = pd.DataFrame(features)
X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=0.8, test_size=0.2, stratify=labels)
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
print("Training model...")
clf = DecisionTreeClassifier(criterion='entropy', max_depth=12, min_samples_leaf=1, splitter="best")
trained_model = clf.fit(X_train, y_train)
print("Score:", trained_model.score(X_train, y_train))
print("Predicting...")
y_pred = clf.predict(X_test)
print("Computing performance metrics...")
results = confusion_matrix(y_test, y_pred)
error = zero_one_loss(y_test, y_pred)
print(f"error是{error}")
recall=X_test/(X_test+X_train)
print(recall)
list_diag = np.diag(results)
list_raw_sum = np.sum(results, axis=1)
print("Predict accuracy of the decisionTree: ", np.mean(list_diag) / np.mean(list_raw_sum))
print("Testing model...")
clf = DecisionTreeClassifier(criterion='entropy', max_depth=12, min_samples_leaf=1, splitter="best")
tested_model = clf.fit(X_test, y_test)
print("Score:", trained_model.score(X_test, y_test))
print("Predicting...")
y_pred = clf.predict(X_train)
print("Computing performance metrics...")
results = confusion_matrix(y_train, y_pred)
error = zero_one_loss(y_train, y_pred)  # 评估模型，评估出错误率
print(f'error是{error}')
list_diag = np.diag(results)
list_raw_sum = np.sum(results, axis=1)
print("Predict accuracy of the decisionTree: ", np.mean(list_diag) / np.mean(list_raw_sum))
import numpy as np
gini_thresholds = np.linspace(0, 0.5, 20)
# entropy_thresholds = np.linespace(0, 1, 50)
parameters = {'splitter': (['best'])#random
    , 'criterion': (["gini"])#"entropy")
    , "max_depth": [*range(10,20 )]
    , 'min_samples_leaf': [*range(1, 20, 5)]
    , 'min_impurity_decrease': [*gini_thresholds]}
clf = DecisionTreeClassifier(random_state=25)  # 实例化决策树
GS: GridSearchCV = GridSearchCV(clf, parameters,cv=10)  # 实例化网格搜索，cv指的是交叉验证
GS.fit(X_train, y_train)
print("你好！")
print(GS.best_params_)  # 从我们输入的参数和参数取值的列表中，返回最佳组合
print(GS.best_score_)  # 网格搜索后的模型的评判标准
# # 调参
# import matplotlib.pyplot as plt
# test = []
# for i in range(20):
#     clf = DecisionTreeClassifier(max_depth=i + 1
#
#                                  , criterion="entropy"
#
#                                  , random_state=30
#
#                                  , splitter="random"
#
#                                  )
#     clf = clf.fit(X_train, y_train)
#     score = clf.score(X_test, y_test)
#     test.append(score)
# plt.plot(range(1, 21), test, color="red", label="max_depth")
# plt.legend()
# plt.show()
