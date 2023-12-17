import pandas
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import metrics

dataset = pandas.read_csv("results.csv")

dataset = dataset.sample(frac=1)

print(dataset)

target = dataset.iloc[:, 0].values
data = dataset.iloc[:, 1:7].values

kfold_object = KFold(n_splits=4)
kfold_object.get_n_splits(data)

# print(kfold_object)

i = 0
for train_index, test_index in kfold_object.split(data):
    i = i + 1
    print("Round:", str(i))
    print("Training index: ")
    print(train_index)
    print("Testing index: ")
    print(test_index)

    data_train = data[train_index]
    target_train = target[train_index]
    data_test = data[test_index]
    target_test = target[test_index]

    machine = linear_model.LinearRegression()
    machine.fit(data_train, target_train)

    prediction = machine.predict(data_test)

    r2 = metrics.r2_score(target_test, prediction)
    print("R square score: ", r2)
    print("\n\n")

#Logistic Regression

target_log = dataset.iloc[:, 0].values
data_log = dataset.iloc[:, 1:7].values


i = 0
for train_index_log, test_index_log in kfold_object.split(data):
    i = i + 1
    print("Round:", str(i))
    print("Training index: ")
    print(train_index_log)
    print("Testing index: ")
    print(test_index_log)

    data_train_log = data[train_index_log]
    target_train_log = target[train_index_log]
    data_test_log = data[test_index_log]
    target_test_log = target[test_index_log]

    machine_log = linear_model.LogisticRegression()
    machine_log.fit(data_train_log, target_train_log)

    prediction_log = machine_log.predict(data_test_log)

    accuracy_score = metrics.accuracy_score(target_test_log, prediction_log)
    print("Accuracy score: ", accuracy_score)

    confusion_matrix_log = metrics.confusion_matrix(target_test_log, prediction_log)
    print("Confusion matrix: ")
    print(confusion_matrix_log)

    print("\n\n")