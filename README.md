# Prediction-using-unsupervised-ML
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')


iris_data = pd.read_csv('Iris.csv')
iris_data.head()
   

   

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

iris_data['Species'] = le.fit_transform(iris_data['Species'])
iris_data['Species']

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

x = iris_data.iloc[:, [0, 1, 2, 3, 4]].values
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(13,8))
plt.plot(range(1, 11), wcss,marker='o')
plt.title('The elbow method',size=15)
plt.xlabel('Number of clusters',size=12)
plt.ylabel('WCSS',size=12) #within cluster sum of squares
plt.show()

kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
predictions = kmeans.fit_predict(x)

predictions

kmeans.cluster_centers_

Features = ['Sepal Length','Sepal Width','Petal Length','Petal Width']
plt.figure(figsize=(18,14))
for i in range(1,5):
    plt.subplot(2,2,i)
    plt.scatter(x[predictions == 0,0], x[predictions == 0,i], s=50, c = '#c718f2', label = 'Iris-setosa' )
    plt.scatter(x[predictions == 1,0], x[predictions == 1,i], s=50, c = '#2140ed', label = 'Iris-vergiscolor' )
    plt.scatter(x[predictions == 2,0], x[predictions == 2,i], s=50, c = '#2cb510', label = 'Iris-virginica' )
    #centroids of the clusters
    plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,i], s = 120, c = 'red', label = 'Centroids')
    plt.title(Features[i-1],size=16)
    plt.xlabel('Id',size=12)
    plt.ylabel(iris_data.columns[i],size=12)
    plt.legend()
plt.suptitle('Clusters w.r.t Features',fontsize=20)

