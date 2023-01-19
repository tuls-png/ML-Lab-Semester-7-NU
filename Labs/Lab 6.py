import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
print("data",data1)
print('---------------------------------------')

X_train, X_test, y_train, y_test = train_test_split(data1, labels, test_size = 0.2, random_state = 0)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


pca = PCA(n_components=4)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score



# k_folds = KFold(n_splits = 10)
#
# clf = DecisionTreeClassifier(random_state=42, max_depth=2)
#
# scores = cross_val_score(clf, X_train, y_train, cv = k_folds)
#
# print("Cross Validation Scores: ", scores)
# print("Average CV Score: ", scores.mean())
# print("Standard Deviation CV Score: ", np.std(scores))
#
# print('////////////////////////////////')


from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                               stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1,
                               stop=X_set[:, 1].max() + 1, step=0.01))



plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('(Training set)')
plt.xlabel('PC1')  # for Xlabel
plt.ylabel('PC2')  # for Ylabel
plt.legend()  # to show legend

# show scatter plot
plt.show()
from matplotlib.colors import ListedColormap

X_set, y_set = X_test, y_test

X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                               stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1,
                               stop=X_set[:, 1].max() + 1, step=0.01))


plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

# title for scatter plot
plt.title('(Test set)')
plt.xlabel('PC1')  # for Xlabel
plt.ylabel('PC2')  # for Ylabel
plt.legend()

# show scatter plot
plt.show()
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score