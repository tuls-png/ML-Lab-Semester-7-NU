import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
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

data =data.replace("Besni", "0")
data = data.replace("Kecimen", "1")
print(labels)

X_train, X_test, y_train, y_test = train_test_split(data1, labels, test_size = 0.2, random_state = 0)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


pca = PCA(n_components=2)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_
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

C = [1, 10, 100, 1000]

for i in range(2, 5):# import SVC model
    for j in range(len(C)):
        classifier = SVC(kernel = 'poly', random_state = 0, degree=i, C=C[j] ) # creat model's object
        classifier.fit(X_train, y_train) # fits the model to the training data

        y_predict = classifier.predict(X_test)
        from sklearn.metrics import confusion_matrix, accuracy_score
        con_matrx = confusion_matrix(y_test, y_predict)
        print(i, C[j])
        print(con_matrx)
        print(accuracy_score(y_test, y_predict))
        # ytrain = []
        # for i in range(len(y_train)):
        #     if y_train[i]=="Besni":
        #         ytrain.append(0)
        #     else:
        #         ytrain.append((1))
        # ytrain1=np.array(ytrain)
        # plot_decision_regions(X_train, ytrain1, clf=classifier, legend=2)
        #
        # # Adding axes annotations
        # plt.xlabel('Data')
        # plt.ylabel('Label')
        # plt.title('SVM on Raisin')
        # plt.show()
        bb=[]

        from mlxtend.plotting import plot_decision_regions
        import matplotlib.pyplot as plt


        plot_decision_regions(X_train, y_train, clf=classifier, legend=2)

        # Adding axes annotations
        plt.xlabel('data')
        plt.ylabel('label')
        plt.title('SVM on Raisin')
        plt.show()


print("------------------------")

lol = [0.5, 1, 2, 4]
for i in range(len(lol)):# import SVC model
    for j in range(len(C)):
        classifier = SVC(kernel = 'rbf', random_state = 0, gamma = lol[i] , C=C[j] ) # creat model's object
        classifier.fit(X_train, y_train) # fits the model to the training data

        y_predict = classifier.predict(X_test)
        from sklearn.metrics import confusion_matrix, accuracy_score
        con_matrx = confusion_matrix(y_test, y_predict)
        print(lol[i], C[j])
        print(con_matrx)
        print(accuracy_score(y_test, y_predict))
