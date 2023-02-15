
def writeData(file):
     raw_data = pd.read_csv(file, header=None,low_memory=False)
     return raw_data
def mergeData():
    monday = writeData("MachineLearning1\MachineLearningCSV\Monday-WorkingHours.pcap_ISCX.csv")
    monday = monday.drop([0])
    friday1 = writeData("MachineLearning1\MachineLearningCSV\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    friday1 = friday1.drop([0])
    friday2 = writeData("MachineLearning1\MachineLearningCSV\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
    friday2 = friday2.drop([0])
    friday3 = writeData("MachineLearning1\MachineLearningCSV\Friday-WorkingHours-Morning.pcap_ISCX.csv")
    friday3 = friday3.drop([0])
    thursday1 = writeData("MachineLearning1\MachineLearningCSV\Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
    thursday1 = thursday1.drop([0])
    thursday2 = writeData("MachineLearning1\MachineLearningCSV\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
    thursday2 = thursday2.drop([0])
    tuesday = writeData("MachineLearning1\MachineLearningCSV\Tuesday-WorkingHours.pcap_ISCX.csv")
    tuesday = tuesday.drop([0])
    wednesday = writeData("MachineLearning1\MachineLearningCSV\Wednesday-workingHours.pcap_ISCX.csv")
    wednesday = wednesday.drop([0])
    frame = [monday, friday1, friday2, friday3, thursday1, thursday2, tuesday, wednesday]
    # 合并数据
    result = pd.concat(frame)
    list = clearDirtyData(result)
    result = result.drop(list)
    return result

def clearDirtyData(df):
    dropList = df[(df[14]=="Nan")|(df[15]=="Infinity")].index.tolist()
    return dropList
raw_data=mergeData()
file = 'MachineLearning1/MachineLearningCSV/total.csv'
raw_data.to_csv(file, index=False, header=False)
last_column_index = raw_data.shape[1] - 1
print(raw_data[last_column_index].value_counts())
import pandas as pd
def writeData(file):
    print("Loading raw data...")
    raw_data = pd.read_csv(file, header=None, low_memory=False)
    return raw_data
def lookData(raw_data):
    last_column_index = raw_data.shape[1] - 1
    print(raw_data[last_column_index].value_counts())
    labels = raw_data.iloc[:, raw_data.shape[1] - 1:]
    labels = labels.values.ravel()
    label_set = set(labels)
    return label_set
lookData(raw_data)
def saveData(lists, file):
    label_set = lookData(raw_data)
    label_list = list(label_set)
    for i in range(0, len(lists)):
        save = pd.DataFrame(lists[i])
        file1 = file + label_list[i] + '.csv'
        save.to_csv(file1, index=False, header=False)
def separateData(raw_data):
    lists = raw_data.values.tolist()
    temp_lists = []
    for i in range(0, 15):
        temp_lists.append([])
    label_set = lookData(raw_data)
    label_list = list(label_set)
    for i in range(0, len(lists)):
        data_index = label_list.index(lists[i][len(lists[0]) - 1])
        temp_lists[data_index].append(lists[i])
        #if i % 5000 == 0:
            #print(i)
    saveData(temp_lists, 'MachineLearning1/MachineLearningCSV/expendData/')
    return temp_lists
lists = separateData(raw_data)
# 将lists分批保存到file文件路径下
def expendData(lists):
    totall_list = []
    for i in range(0, len(lists)):
        while len(lists[i]) < 5000:
            lists[i].extend(lists[i])
        #print(i)
        totall_list.extend(lists[i])
    saveData(lists, 'MachineLearning1/MachineLearningCSV/expendData/')
    save = pd.DataFrame(totall_list)
    file = 'MachineLearning1/MachineLearningCSV/expendData/totall_extend.csv'
    save.to_csv(file, index=False, header=False)
file = 'MachineLearning1/MachineLearningCSV/total.csv'
raw_data = writeData(file)
expendData(lists)
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn import preprocessing
# 加载数据
raw_data_filename = "MachineLearning1/MachineLearningCSV/expendData/totall_extend.csv"
print("Loading raw data...")
raw_data = pd.read_csv(raw_data_filename, header=None, low_memory=False)
raw_data = raw_data.sample(frac=0.03)
last_column_index = raw_data.shape[1] - 1
print("print data labels:")
print(raw_data[last_column_index].value_counts())

# 将非数值型的数据转换为数值型数据
# print("Transforming data...")
raw_data[last_column_index], attacks = pd.factorize(raw_data[last_column_index], sort=True)
features = raw_data.iloc[:, :raw_data.shape[1] - 1]
labels = raw_data.iloc[:, raw_data.shape[1] - 1:]#标签
features = preprocessing.scale(features)
features = pd.DataFrame(features)
labels = labels.values.ravel()
df = pd.DataFrame(features)
X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=0.8, test_size=0.2, stratify=labels)
# print("X_train,y_train:", X_train.shape, y_train.shape)
# print("X_test,y_test:", X_test.shape, y_test.shape)
#训练和测试：
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
error = zero_one_loss(y_train, y_pred)
list_diag = np.diag(results)
list_raw_sum = np.sum(results, axis=1)
print("Predict accuracy of the decisionTree: ", np.mean(list_diag) / np.mean(list_raw_sum))
import numpy as np
gini_thresholds = np.linspace(0, 0.5, 20)
# entropy_thresholds = np.linespace(0, 1, 50)
parameters = {'splitter': ('best', 'random')
    , 'criterion': ("gini", "entropy")
    , "max_depth": [*range(1, 5)]
    , 'min_samples_leaf': [*range(1, 20, 5)]
    , 'min_impurity_decrease': [*np.linspace(0, 0.5, 20)]
              }
clf = DecisionTreeClassifier(random_state=25)
GS = GridSearchCV(clf, parameters, cv=10)
GS.fit(X_train, y_train)
print(GS.params_)
print(GS.score_)

