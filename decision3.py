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
raw_data = raw_data.sample(frac=0.8)
last_column_index = raw_data.shape[1] - 1
print("print data labels:")
print(raw_data[last_column_index].value_counts())
# print("Transforming data...")
raw_data[last_column_index], attacks = pd.factorize(raw_data[last_column_index], sort=True)
# 对原始数据进行切片，分离出特征和标签，第1~78列是特征，第79列是标签
features = raw_data.iloc[:, :raw_data.shape[1] - 1]  # pandas中的iloc切片是完全基于位置的索引
labels = raw_data.iloc[:, raw_data.shape[1] - 1:]
# 特征数据标准化，这一步是可选项
features = preprocessing.scale(features)
features = pd.DataFrame(features)

# 将多维的标签转为一维的数组
labels = labels.values.ravel()

# 将数据分为训练集和测试集,并打印维数
df = pd.DataFrame(features)
X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=0.8, test_size=0.2, stratify=labels)

# print("X_train,y_train:", X_train.shape, y_train.shape)
# print("X_test,y_test:", X_test.shape, y_test.shape)

# 训练和测试：
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss

# 训练模型的搭建
print("Training model...")
clf = DecisionTreeClassifier(criterion='entropy', max_depth=19, min_samples_leaf=2, min_samples_split=2,splitter="best")
trained_model = clf.fit(X_train, y_train)
print("Score:", trained_model.score(X_train, y_train))#训练集的分数
# 预测
print("Predicting...")
y_pred = clf.predict(X_test)
print("Computing performance metrics...")
results = confusion_matrix(y_test, y_pred)
error = zero_one_loss(y_test, y_pred)
#print(error)
# 根据混淆矩阵求预测精度
list_diag = np.diag(results)
list_raw_sum = np.sum(results, axis=1)
print("Predict accuracy of the decisionTree: ", np.mean(list_diag) / np.mean(list_raw_sum))
from sklearn.metrics import accuracy_score
# score=accuracy_score(y_pred,y_test)
# print(f'{score}is:')这两行和上面那个效果是一样的
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
#print(error)
# 根据混淆矩阵求预测精度
list_diag = np.diag(results)
list_raw_sum = np.sum(results, axis=1)
print("Predict accuracy of the decisionTree: ", np.mean(list_diag) / np.mean(list_raw_sum))
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
# DT = DecisionTreeClassifier(criterion = 'entropy',random_state = 30)
# score2 = cross_val_score(DT,X_train,y_train,cv=10).mean()
# print(f'熵得分：{score2}')
#第二得到最优深度：
# import matplotlib.pyplot as plt
# test = []
# for i in range(50):
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
# plt.plot(range(1, 51), test, color="red", label="max_depth")
# plt.legend()
# plt.show()

#第三单独看看min_samples_split的变化趋势
# ScoreAll = []
# for i in range(2,25):
#     DT = DecisionTreeClassifier(max_depth = 19,min_samples_split = i,random_state = 30)#最优深度是19来着的
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
#     DT = DecisionTreeClassifier(min_samples_leaf = i,max_depth = 19,random_state = 30)#min_samples_split =2)
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

#检查是否过拟合：
train = []
test = []
for i in range(50):
    clf =DecisionTreeClassifier(criterion='gini',
                                     random_state=None,
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     max_depth=i+1)
    clf= clf.fit(X_train,y_train)
    score_tr = clf.score(X_train,y_train)
    score_te = clf.score(X_test,y_test)
    train.append(score_tr)
    test.append(score_te)
print(max(test))
plt.plot(range(1,51),train,color='red',label='train')
plt.plot(range(1,51),test,color='blue',label='test')
plt.xticks(range(1,51,5))
#plt.figure(figsize=[18,7])# 坐标细分
plt.legend()
plt.show()
plt.ylabel('准确率',fontsize=8,labelpad=8)
plt.xlabel('决策树深度',fontsize=8,labelpad=8)

#后剪枝rep算法：
#REP剪枝函数：

# class TreeNode:
#     def __init__(self, val, fea_name=None, fea_c=None):
#         self.left = None
#         self.right = None
#         self.val = round(val,2)
#         self.fea_name = fea_name
#         self.fea_c = fea_c if fea_c is None else round(fea_c,2)
# #生成剪枝后的树：
#     def sub_tree(tree, num): # 返回后序剪枝得到的子树
#         stack = [(False, tree)]
#         while stack:
#             flag, t = stack.pop()
#             if not t:continue
#             if flag:
#                 if t.left or t.right:
#                     if num==0:
#                         t.left = None
#                         t.right = None
#                         return tree
#                     else:
#                         num -= 1
#             else:
#                 stack.append((True, t))
#                 stack.append((False, t.right))
#                 stack.append((False, t.left))
#         return tree
#
#
#     def prune_tree(self, X_test, y_test):
#         mid_num = mid_leaf_num(self.tree)
#         i = 0
#         while i<mid_num:
#             temp_tree = sub_tree(self.tree, i)
#             if self.ifmore(temp_tree, X_test, y_test):
#                 i = 0
#                 mid_num -= 1
#             else:
#                 i += 1
#
#     # 计算中间节点个数：
#     def mid_leaf_num(tree):
#         if not tree or (not tree.left and not tree.right):
#             return 0
#         return 1 + mid_leaf_num(tree.left) + mid_leaf_num(tree.right)
#
#     # 效果比较函数：
#     def ifmore(self, temp_tree, X_test, y_test):
#         orig_ = []
#         temp_ = []
#         for i in range(len(X_test)):
#             orig_.append(self.check(self.tree, X_test[i]))
#             temp_.append(self.check(temp_tree, X_test[i]))
#         orig_sum = sum(np.power(np.array(orig_) - y_test, 2))
#         temp_sum = sum(np.power(np.array(temp_) - y_test, 2))
#         if orig_sum > temp_sum:  # and (orig_sum-temp_sum)/orig_sum>0.0001:
#             self.tree = temp_tree
#             return True
#         else:
#             return False
#
# train_size = len(X_train)//4
# X_train_test, y_train_test = X_train[:train_size], y_train[:train_size]
# X_train_train, y_train_train = X_train[train_size:], y_train[train_size:]
# print('不剪枝')
# clf.fit(X_train, y_train)
# predict_y = clf.predict(X_test)
# pre_error = sum(np.power(y_test-predict_y,2))
# print('误差为：', pre_error)#' 节点数：', clf.node_num())
#
# print('有剪枝')
# clf.fit(X_train_train, y_train_train)
# # x=clf.prune_tree(self,X_train_test, y_train_test)
# x=DecisionTreeClassifier.prune_tree(self,X_train_test, y_train_test)
# predict_y = clf.predict(X_test)
# pre_error = sum(np.power(y_test-predict_y,2))
# print('误差为：', pre_error,' 节点数：')# clf.node_num())
#


















