import pandas as pd
import matplotlib.pyplot as plt
import numpy
from sklearn.model_selection import train_test_split

data = pd.read_csv('Breast Cancer Dataset/breast-cancer-wisconsin.data', sep=",")
data.replace('?', 0, inplace=True)
print('DATASET')
print(data)
print('---------------------')
data.drop(["1000025"], axis=1, inplace=True)
x = data.drop(['2.1'], axis=1)
y = data['2.1']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train, y_train)
GaussianNB()
print("Naive Bayes score: ", nb.score(x_test, y_test))
print('---------------------------------')
E_d=1-nb.score(x_test, y_test)
print("Error rate: ",E_d)

#dropping feature 9
feature_cols=['5', '1', '1.1', '1.2', '2', '1.3', '3', '1.4']
x=data[feature_cols]
y=data['2.1']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33,random_state = 42)
nb = GaussianNB()
nb.fit(x_train, y_train)
E_d=1-nb.score(x_test, y_test)
print("Naive Bayes score by dropping feature 9: ", nb.score(x_test, y_test))

#dropping feature 8
feature_cols=['5', '1', '1.1', '1.2', '2', '1.3', '3', '1.5']
x=data[feature_cols]
y=data['2.1']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33,random_state = 42)
nb = GaussianNB()
nb.fit(x_train, y_train)
E_d=1-nb.score(x_test, y_test)
print("Naive Bayes score by dropping feature 8: ", nb.score(x_test, y_test))

#dropping feature 7
feature_cols=['5', '1', '1.1', '1.2', '2', '1.3', '1.4', '1.5']
x=data[feature_cols]
y=data['2.1']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33,random_state = 42)
nb = GaussianNB()
nb.fit(x_train, y_train)
E_d=1-nb.score(x_test, y_test)
print("Naive Bayes score by dropping feature 7:", nb.score(x_test, y_test))

#dropping feature 6
feature_cols=['5', '1', '1.1', '1.2', '2', '3', '1.4', '1.5']
x=data[feature_cols]
y=data['2.1']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33,random_state = 42)
nb = GaussianNB()
nb.fit(x_train, y_train)
E_d=1-nb.score(x_test, y_test)
print("Naive Bayes score by dropping feature 7: ", nb.score(x_test, y_test))

#dropping feature 5
feature_cols=['5', '1', '1.1', '1.2', '1.3', '3', '1.4', '1.5']
x=data[feature_cols]
y=data['2.1']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33,random_state = 42)
nb = GaussianNB()
nb.fit(x_train, y_train)
E_d=1-nb.score(x_test, y_test)
print("Naive Bayes score by dropping feature 7: ", nb.score(x_test, y_test))

#dropping feature 4
feature_cols=['5', '1', '1.1', '2', '1.3', '3', '1.4', '1.5']
x=data[feature_cols]
y=data['2.1']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33,random_state = 42)
nb = GaussianNB()
nb.fit(x_train, y_train)
E_d=1-nb.score(x_test, y_test)
print("Naive Bayes score by dropping feature 7: ", nb.score(x_test, y_test))

#dropping feature 3
feature_cols=['5', '1', '1.2', '2', '1.3', '3', '1.4', '1.5']
x=data[feature_cols]
y=data['2.1']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33,random_state = 42)
nb = GaussianNB()
nb.fit(x_train, y_train)
E_d=1-nb.score(x_test, y_test)
print("Naive Bayes score by dropping feature 7: ", nb.score(x_test, y_test))

#dropping feature 2
feature_cols=['5', '1.1', '1.2', '2', '1.3', '3', '1.4', '1.5']
x=data[feature_cols]
y=data['2.1']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33,random_state = 42)
nb = GaussianNB()
nb.fit(x_train, y_train)
E_d=1-nb.score(x_test, y_test)
print("Naive Bayes score by dropping feature 7: ", nb.score(x_test, y_test))

#dropping feature 1
feature_cols=['1', '1.1', '1.2', '2', '1.3', '3', '1.4', '1.5']
x=data[feature_cols]
y=data['2.1']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33,random_state = 42)
nb = GaussianNB()
nb.fit(x_train, y_train)
E_d=1-nb.score(x_test, y_test)
print("Naive Bayes score by dropping feature 7: ", nb.score(x_test, y_test))

#dropping feature 9 and 8
feature_cols=['5', '1', '1.1', '1.2', '2', '1.3', '3']
x=data[feature_cols]
y=data['2.1']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33,random_state = 42)
nb = GaussianNB()
nb.fit(x_train, y_train)
E_d=1-nb.score(x_test, y_test)
print("Naive Bayes score by dropping feature 9 and 8: ", nb.score(x_test, y_test))

#dropping feature 9 and 7
feature_cols=['5', '1', '1.1', '1.2', '2', '1.3', '1.4']
x=data[feature_cols]
y=data['2.1']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33,random_state = 42)
nb = GaussianNB()
nb.fit(x_train, y_train)
E_d=1-nb.score(x_test, y_test)
print("Naive Bayes score by dropping feature 9 and 7: ", nb.score(x_test, y_test))

#dropping feature 9 and 6
feature_cols=['5', '1', '1.1', '1.2', '2', '3', '1.4']
x=data[feature_cols]
y=data['2.1']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33,random_state = 42)
nb = GaussianNB()
nb.fit(x_train, y_train)
E_d=1-nb.score(x_test, y_test)
print("Naive Bayes score by dropping feature 9 and 6: ", nb.score(x_test, y_test))

#dropping feature 9 and 5
feature_cols=['5', '1', '1.1', '1.2', '1.3', '3', '1.4']
x=data[feature_cols]
y=data['2.1']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33,random_state = 42)
nb = GaussianNB()
nb.fit(x_train, y_train)
E_d=1-nb.score(x_test, y_test)
print("Naive Bayes score by dropping feature 9 and 5: ", nb.score(x_test, y_test))

#dropping feature 9 and 4
feature_cols=['5', '1', '1.1', '2', '1.3', '3', '1.4']
x=data[feature_cols]
y=data['2.1']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33,random_state = 42)
nb = GaussianNB()
nb.fit(x_train, y_train)
E_d=1-nb.score(x_test, y_test)
print("Naive Bayes score by dropping feature 9 and 4: ", nb.score(x_test, y_test))

#dropping feature 9 and 3
feature_cols=['5', '1', '1.2', '2', '1.3', '3', '1.4']
x=data[feature_cols]
y=data['2.1']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33,random_state = 42)
nb = GaussianNB()
nb.fit(x_train, y_train)
E_d=1-nb.score(x_test, y_test)
print("Naive Bayes score by dropping feature 9 and 3: ", nb.score(x_test, y_test))

#dropping feature 9 and 2
feature_cols=['5', '1.1', '1.2', '2', '1.3', '3', '1.4']
x=data[feature_cols]
y=data['2.1']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33,random_state = 42)
nb = GaussianNB()
nb.fit(x_train, y_train)
E_d=1-nb.score(x_test, y_test)
print("Naive Bayes score by dropping feature 9 and 2: ", nb.score(x_test, y_test))

#dropping feature 9 and 1
feature_cols=['1', '1.1', '1.2', '2', '1.3', '3', '1.4']
x=data[feature_cols]
y=data['2.1']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33,random_state = 42)
nb = GaussianNB()
nb.fit(x_train, y_train)
E_d=1-nb.score(x_test, y_test)
print("Naive Bayes score by dropping feature 9 and 1: ", nb.score(x_test, y_test))