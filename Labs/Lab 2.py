import pandas as pd
from sklearn. model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
data = pd.read_csv('Breast Cancer Dataset/breast-cancer-wisconsin.data', sep=",")
data.replace('?', 0, inplace=True)
print('DATASET')
print(data)
print('---------------------')
data.drop(["1000025"],axis=1,inplace=True)
X = data.drop(['2.1'], axis=1)
y = data['2.1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print('X_train')
print(X_train)
print('-----------')
print('X_test')
print(X_test)
print('-----------')

#building a univariate decision tree

def train_using_gini(X_train, X_test, y_train):

    clf_gini = DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=None, min_samples_leaf=5)
    clf_gini.fit(X_train, y_train)
    pred = clf_gini.predict(X_test)
    score = accuracy_score(y_test, pred)
    print('Accuracy Score', score)
    depth=clf_gini.tree_.max_depth
    print('Depth of tree',depth)
    a=[]
    d=[]
    for i in range(2, depth+1):
        clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=i, min_samples_leaf=5)
        clf_gini.fit(X_train, y_train)
        pred = clf_gini.predict(X_test)
        score = accuracy_score(y_test, pred)
        print('---')
        print(f'Accuracy Score {i}:', score)
        a.append(score)
        depth = clf_gini.tree_.max_depth
        print(f'Depth of tree {i}:', depth)
        d.append(depth)
    print(a)
    print(d)
    plt.plot(d, a)
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')
    plt.show()
train_using_gini(X_train,X_test,y_train)
