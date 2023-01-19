from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
data =pd.read_csv('seeds_dataset.txt', sep='\s+')
print('DATASET')
print(data)
print('---------------------')
data = data.iloc[:,:-1]
print(data)
distortions = []
clusters = []

K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(data)
    clusters.append(kmeanModel)
    distortions.append(kmeanModel.inertia_)
print(clusters)
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

for i in range(1, 9):

    print("Silhouette score:", silhouette_score(data, clusters[i].predict(data)))