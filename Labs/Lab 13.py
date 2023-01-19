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
print(X)
y = df["Absenteeism time in hours"]
print (y)

from sklearn.model_selection import train_test_split
# i.e. 70 % training dataset and 30 % test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

clf = RandomForestClassifier(n_estimators=100)
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)

# performing predictions on the test dataset
y_pred = clf.predict(X_test)

# metrics are used to find accuracy or error
from sklearn import metrics

print()

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

accuracy=[]
features=[]
for i in range(2,20):
    features.append(i)
    clf = RandomForestClassifier(n_estimators=100, max_features=i)
    # Training the model on the training dataset
    # fit function is used to train the model using the training sets as parameters
    clf.fit(X_train, y_train)
    # performing predictions on the test dataset
    y_pred = clf.predict(X_test)
    # metrics are used to find accuracy or error
    from sklearn import metrics
    print()
    # using metrics module for accuracy calculation
    acc=metrics.accuracy_score(y_test, y_pred)
    accuracy.append(acc)
    print(f"ACCURACY OF THE MODEL with {i} FEATURES: ", acc )

plt.plot(features, accuracy)
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Features')
plt.show()

grid_space={
    'criterion': ['gini', 'entropy'],
    'max_depth' : range(2,5,1),
    'max_features' : list(range(2,20,1))
}
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(RandomForestClassifier(),param_grid=grid_space,cv=3,scoring='accuracy')
model_grid = grid.fit(X_train,y_train)
print('Best hyperparameters are: '+str(model_grid.best_params_))
print('Best score is: '+str(model_grid.best_score_))

rand_clf1 = RandomForestClassifier(criterion= 'gini',
 max_features = 2,
 max_depth =  2, oob_score=True
 )

rand_clf1.fit(X_train,y_train)
y_pred = clf.predict(X_test)
# metrics are used to find accuracy or error
from sklearn import metrics
print()
# using metrics module for accuracy calculation
acc=metrics.accuracy_score(y_test, y_pred)
accuracy.append(acc)
print(f"ACCURACY OF THE MODEL AFTER OOB SCORE: ", acc )
ff=['Reason for absence','Month of absence','Day of the week','Seasons','Transportation expense','Distance from Residence to Work','Service time','Age','Hit target','Disciplinary failure','Education','Son','Social drinker','Social smoker','Pet','Weight','Height','Body mass index']

pp=rand_clf1.feature_importances_
print(pp)

