import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
# reading the given csv file
# and creating dataframe

file = open("ethylene_CO.txt")
CO_lines = file.readlines()
CO_lines.pop(0)
data_list = []
def formater(lines):
  for line in lines:
    txt = line.split(" ")
    dummy = []
    for t in txt:
      if t != '':
        dummy.append(t)
        dummy[-1] = dummy[-1].replace("\n", "")
    data_list.append(dummy)
formater(CO_lines)
data= pd.DataFrame(data_list, columns =  ['Time(sec)',
 'CO con: (ppm)',
 'Ethylene con: (ppm',
 'TGS2602-1',
 'TGS2602-2',
 'TGS2600-1',
 'TGS2600-2',
 'TGS2610-1',
 'TGS2610-2',
 'TGS2620-1',
 'TGS2620-2',
 'TGS2602-3',
 'TGS2602-4',
 'TGS2600-3',
 'TGS2600-4',
 'TGS2610-3',
 'TGS2610-4',
 'TGS2620-3',
 'TGS2620-4',""])
data = data.drop([""],axis = 1)
data.head()
data.info()
data.to_csv("ethylene_CO.csv")

df = pd.read_csv("ethylene_CO.csv")
#print(dataframe1.isnull())
#fig = plt.figure(figsize=(4,4))
#ax = fig.add_subplot(111, projection="3d")
#ax.scatter(dataframe1['CO conc (ppm)'], dataframe1['Ethylene conc (ppm)'], dataframe1['sensor readings (16 channels)'])

fig = plt.figure(figsize=(4, 4))
ax = Axes3D(fig)
y = df.iloc[:,1]
x = df.iloc[:,2]
z = df.iloc[:,3]
#c = df['grp']
ax.scatter(x,y,z,cmap='coolwarm')
plt.title('First 3 Principal Components')
ax.set_ylabel('PC2')
ax.set_xlabel('PC1')
ax.set_zlabel('PC3')
plt.legend()
plt.show()