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

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix : \n", cm)
print(" ")
from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test, y_pred))
print(" ")
featureimportance=[]
features=[]
importance = classifier.coef_[0]
# summarize feature importance
print('Feature Importance:')
for i,v in enumerate(importance):
    featureimportance.append(abs(v))
    features.append(i+1)
    print(i+1," : ", abs(v))
print(" ")
res = {features[i]: featureimportance[i] for i in range(len(features))}


sorted_dict = sorted(
    res.items(),
    key=lambda kv: kv[1], reverse=True)
sortedfi=[]
# Print sorted dictionary
print("Sorted Feature Importance:",
      sorted_dict)
print(" ")
for i in range(len(sorted_dict)):
    sortedfi.append(sorted_dict[i][0])

print("Features in terms of importance:")
print(sortedfi)
print(" ")
# plt.bar([x for x in range(len(importance))], importance)
# plt.show()

from sklearn.feature_selection import mutual_info_classif as MIC
features1=[]
mi_score = MIC(data1, labels)
mutualinformation=[]
sortedmi=[]
print('Mutual Information List:')
for i,v in enumerate(mi_score):
    mutualinformation.append(v)
    features1.append(i + 1)
    print(i + 1, " : ", v)
res1 = {features1[i]: mutualinformation[i] for i in range(len(features1))}
sorted_dict1 = sorted(
    res1.items(),
    key=lambda kv: kv[1], reverse=True)
sortedfi1=[]
# Print sorted dictionary
print("Sorted Mutual Information:",
      sorted_dict1)
print(" ")

for i in range(len(sorted_dict1)):
    sortedfi1.append(sorted_dict1[i][0])

print("Features in terms of Mutual Information:")
print(sortedfi1)
print(" ")

for i in range (len(sortedfi)):
    sortedmi.append(mutualinformation[sortedfi[i]-1])
print(sortedmi)