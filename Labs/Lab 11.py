import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import ward, median, centroid, weighted, average, complete, single, fcluster
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import single, cophenet
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

data =pd.read_csv('Absenteeism_at_work.csv', sep=';')
print('DATASET')
print(data)
print('---------------------')
data = data.iloc[:,1:]
data = data.iloc[:,:-1]
print('DATASET AFTER DROPPING FIRST AND LAST COLUMN')
print(data)
print('---------------------')

X = data[['Reason for absence','Month of absence','Day of the week','Seasons','Transportation expense','Distance from Residence to Work','Service time','Age','Hit target','Disciplinary failure','Education','Son','Social drinker','Social smoker','Pet','Weight','Height','Body mass index']]
plt.figure(figsize=(10,6))
dend = sch.dendrogram(sch.linkage(X, method = 'single'))

# plt.title('Dendrogram - Single Linkage')
# plt.xlabel('Samples')
# plt.ylabel('Euclidean distances')
# # plt.show()
#
# plt.figure(figsize=(10,6))
# dend = sch.dendrogram(sch.linkage(X, method = 'complete'))
# plt.title('Dendrogram - Complete Linkage')
# plt.xlabel('Samples')
# plt.ylabel('Euclidean distances')
# # plt.show()
#
# plt.figure(figsize=(10,6))
# dend = sch.dendrogram(sch.linkage(X, method = 'ward'))
# plt.title('Dendrogram - Ward Linkage')
# plt.xlabel('Samples')
# plt.ylabel('Euclidean distances')
# # plt.show()

from scipy.cluster.hierarchy import single, cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
Y = pdist(X)
Z = linkage(Y,'single')
[c,D] = cophenet(Z,Y)
print("cophenetic correlation coefficient - Single",c)

Z = linkage(Y,'complete')
[c,D] = cophenet(Z,Y)
print("cophenetic correlation coefficient - Complete",c)

Z = linkage(Y,'ward')
[c,D] = cophenet(Z,Y)
print("cophenetic correlation coefficient - Ward",c)

# -------------------------
print('------------------------------')
from sklearn.preprocessing import StandardScaler, normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)
X_normalized = normalize(X_scaled)
X_normalized = pd.DataFrame(X_normalized)

# plt.figure(figsize =(8, 8))
# plt.title('Visualising the data')
# Dendrogram = sch.dendrogram((sch.linkage(X_normalized, method ='single')))
# plt.show()
#
# Dendrogram = sch.dendrogram((sch.linkage(X_normalized, method ='complete')))
# plt.show()
#
# Dendrogram = sch.dendrogram((sch.linkage(X_normalized, method ='ward')))
# plt.show()

Y = pdist(X_normalized)
Z = linkage(Y,'single')
[c,D] = cophenet(Z,Y)
print("cophenetic correlation coefficient - Single",c)

Z = linkage(Y,'complete')
[c,D] = cophenet(Z,Y)
print("cophenetic correlation coefficient - Complete",c)

Z = linkage(Y,'ward')
[c,D] = cophenet(Z,Y)
print("cophenetic correlation coefficient - Ward",c)

