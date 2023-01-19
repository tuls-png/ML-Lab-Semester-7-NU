import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('parkinsons.data', sep=",")
data.replace('?', 0, inplace=True)
print('DATASET')
print(data)
print('---------------------')
data.drop(["name"], axis=1, inplace=True)
print("---------------------")
print(data)
df = pd.DataFrame(data)
df = df.sample(frac = 1)
print('Shuffled Dataset')
print(df)
print('---------------------------------------')
x=df[['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA','NHR','HNR','RPDE','DFA','spread1','spread2','D2','PPE']]
y=df["status"]
print(x)
print(y)
print('---------------------------------------')
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.neural_network import MLPClassifier
m=[2,3,4,5]
l=[0.01, 0.03, 0.05, 0.07, 0.09, 0.11]
accuracy = []
for i in range(len(m)):
    lol = []
    for j in range(len(l)):

        clf = MLPClassifier(hidden_layer_sizes=(m[i],1),
                            random_state=5,
                            verbose=True,
                            learning_rate_init=l[j], learning_rate="invscaling")

        # Fit data onto the model
        clf.fit(X_train,y_train)

        print(f"Hidden Layer: {m[i]}, Learning Rate:{l[j]}")
        ypred = clf.predict(X_test)

        # Import accuracy score
        from sklearn.metrics import accuracy_score

        # Calcuate accuracy
        p = accuracy_score(y_test, ypred)
        lol.append(p)
        print(p)
    print(lol)
    print("--------------------------------------------")
    accuracy.append(lol)
print(accuracy)

plt.plot(l, accuracy[0], color='r', label='two')
plt.plot(l, accuracy[1], color='g', label='three')
plt.plot(l, accuracy[2], color='b', label='four')
plt.plot(l, accuracy[3], color='m', label='five')



# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")


# Adding legend, which helps us recognize the curve according to it's color
plt.legend()

# To load the display window
plt.show()
