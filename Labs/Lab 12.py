import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
df =pd.read_csv('Absenteeism_at_work.csv', sep=';')

df.drop("ID",axis=1,inplace=True)
df['Absenteeism time in hours'] = pd.cut(df['Absenteeism time in hours'], bins=[0, 10, 20, 30, 40, 50, 60, 130], labels=np.arange(7), right=False)

X = df.drop("Absenteeism time in hours",axis=1)
y = df["Absenteeism time in hours"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
acc_list = []
dc = DecisionTreeClassifier(max_depth=1)
print('-----------')
print(f'For 1st depth: ')

for i in range(10,110,10):
    clf = AdaBoostClassifier(base_estimator=dc,algorithm="SAMME",n_estimators=i, random_state=0)
    clf.fit(X_train, y_train)
    acc_list.append(clf.score(X_train, y_train))
print(acc_list)
gg = [10,20,30,40,50,60,70,80,90,100]
plt.plot(gg,acc_list)
plt.show()


dc = DecisionTreeClassifier(max_depth=2)
print('-----------')
print(f'For 2nd depth: ')
acc_list = []
for i in range(10,110,10):
    clf = AdaBoostClassifier(base_estimator=dc,algorithm="SAMME",n_estimators=i, random_state=0)
    clf.fit(X_train, y_train)
    acc_list.append(clf.score(X_train, y_train))
print(acc_list)
gg = [10,20,30,40,50,60,70,80,90,100]
plt.plot(gg,acc_list)
plt.show()



dc = DecisionTreeClassifier(max_depth=3)
print('-----------')
print(f'For 3rd depth: ')
acc_list = []
for i in range(10,110,10):
    clf = AdaBoostClassifier(base_estimator=dc,algorithm="SAMME",n_estimators=i, random_state=0)
    clf.fit(X_train, y_train)
    acc_list.append(clf.score(X_train, y_train))
print(acc_list)
gg = [10,20,30,40,50,60,70,80,90,100]
plt.plot(gg,acc_list)
plt.show()


