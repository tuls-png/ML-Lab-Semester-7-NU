import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_excel('Raisin_Dataset/Raisin_Dataset.xlsx')
print('Original Dataset')
print(data)
print('---------------------------------------')
df = pd.DataFrame(data)
df = df.sample(frac = 1)
print('Shuffled Dataset')
print(df)
print('---------------------------------------')
raw_data = df.values
data1 = raw_data[:, 0:-1]
print(len(data1))
labels = raw_data[:, -1]
print("data",data)
print("labels",labels)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score



k_folds = KFold(n_splits = 10)

clf = DecisionTreeClassifier(random_state=42, max_depth=2)

scores = cross_val_score(clf, data1, labels, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Standard Deviation CV Score: ", np.std(scores))

print('////////////////////////////////')
avgacc=[]
maxdepth=[]
standarddev=[]
for i in range(2,8):
    clf = DecisionTreeClassifier(random_state=42, max_depth=i)

    scores = cross_val_score(clf, data1, labels, cv=k_folds)

    print("Cross Validation Scores: ", scores)
    m=scores.mean()
    print("Average CV Score: ", m)
    avgacc.append(m)
    maxdepth.append(i)
    stdv=np.std(scores)
    print("Standard Deviation CV Score: ", stdv)
    standarddev.append(stdv)

plt.title('Average Accuracy vs Maximum Depth')
plt.plot(maxdepth, avgacc)
plt.xlabel('Depth')
plt.ylabel('Average Accuracy')


plt.errorbar(maxdepth, avgacc, yerr=stdv)
plt.show()
# kf = KFold(n_splits=10)
# for train, test in kf.split(df):
#     print("%s %s"  % (df[train], df[test]))
#     print('---------')
#

# import numpy as np
#
# permuted_indices = np.random.permutation(len(df))
#
# dfs = []
# for i in range(10):
#     dfs.append(df.iloc[permuted_indices[i::10]])
# print(dfs)
# print('---------------------------------------')
# print(dfs[0])