import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import category_scatter
from sklearn.metrics import silhouette_score
# Question 1
df = pd.read_csv("./specs/question_1.csv")
# Initiazlizing kmeans with the required parameters for the dataframe df
kmeans = KMeans(n_clusters = 3, random_state = 0).fit(df)
# Assigning predicition values to a variable clusters
clusters = kmeans.predict(df)
print(kmeans.cluster_centers_)

# Creating a Column cluster in df and assigning the clusters variable value to it
df['cluster'] = clusters

df.to_csv("./output/question_1.csv", index = False)

# Using category_scatter from mlxtend to plot the df with the labels
fig = category_scatter(x = 'x', y = 'y', label_col = 'cluster', data = df, legend_loc='upper right')
plt.grid(True)
#Calculating the centroids of the clusters
centroid = kmeans.cluster_centers_
plt.scatter(centroid[:, 0], centroid[:, 1], c = 'purple', s = 100, alpha = 0.2, label = "centroid")
plt.title('K Means Clustering Q1')
plt.xlabel('X Co - ordinate')
plt.ylabel('Y Co - ordinate')
plt.legend(loc = 'upper right', title = "clusters")

plt.show()

# Question 2
df2 = pd.read_csv("./specs/question_2.csv")

# Removing the columns which are not required
df2 = df2.drop(columns="NAME")
df2 = df2.drop(columns="MANUF")
df2 = df2.drop(columns="TYPE")
df2 = df2.drop(columns="RATING")


# Initializing kmeans with 5 clusters, 5 max runs and 100 max optimization steps
kmeans = KMeans(n_clusters=5, random_state=0, n_init=5, max_iter=100).fit(df2)
clusters1 = kmeans.predict(df2)

# Initializing kmeans with 5 clusters, 100 max runs and 100 max optimization steps
kmeans = KMeans(n_clusters=5, random_state=0, n_init=100, max_iter=100).fit(df2)
clusters2 = kmeans.predict(df2)
labels2 = kmeans.labels_

#calculating the silhouette score to check which configuration is better
print(silhouette_score(df2, labels2, metric='euclidean'))

# Initializing kmeans with 3 clusters, 100 max runs and 100 max optimization steps
kmeans = KMeans(n_clusters=3, random_state=0, n_init=100, max_iter=100).fit(df2)
clusters3 = kmeans.predict(df2)
labels3 = kmeans.labels_
print(silhouette_score(df2, labels3, metric='euclidean'))

df2['config1'] = clusters1
df2['config2'] = clusters2
df2['config3'] = clusters3

df2.to_csv("./output/question_2.csv", index = False)

# Question 3

df4 = pd.read_csv("./specs/question_3.csv")

#Deleting Column ID
df4 = df4.drop(columns='ID')

# Initializing kmeans with 7 clusters, 5 max runs and 100 max optimization steps
kmeans = KMeans(n_clusters=7, random_state=0, n_init=5, max_iter=100).fit(df4)
clusters = kmeans.predict(df4)

# Storing the labels in column kmeans
df4['kmeans'] = clusters

# Plotting the Kmeans Clusters
fig = category_scatter(x = 'x', y = 'y', label_col = 'kmeans', data = df4, legend_loc='upper left')
plt.grid(True)
centroid = kmeans.cluster_centers_
plt.scatter(centroid[:, 0], centroid[:, 1], c = 'purple', s = 100, alpha = 0.2, label = "centroid")
plt.title('K Means Clustering Q3')
plt.xlabel('X Co - ordinate')
plt.ylabel('Y Co - ordinate')
plt.legend(loc = 'upper left', title = " clusters")
plt.show()

#Normalizing the columns
from sklearn import preprocessing

#selecting the columns to normalize
x = df4[['x','y']]
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

#creating a new df and storing the normalized values
df5 = pd.DataFrame(x_scaled)
df5.columns = ['x','y']

#Performing DBSCAN
from sklearn.cluster import DBSCAN

#DBSCAN with 0.04 epsilon value
clustering = DBSCAN(eps=0.04, min_samples=4).fit(df5)
dbscan1 = clustering.labels_

#DBSCAN with 0.08 epsilon value
clustering = DBSCAN(eps=0.08, min_samples=4).fit(df5)
dbscan2 = clustering.labels_

# Storing all the values in new columns for testing
df5['kmeans'] = df4['kmeans']
df5['dbscan1'] = dbscan1
df5['dbscan2'] = dbscan2
df5.to_csv("./output/question_3.csv", index= False)

fig = category_scatter(x = 'x', y = 'y', label_col = 'dbscan1', data = df5, legend_loc='upper left')
plt.title('DBSCAN with 0.04 eps ')
plt.xlabel('X Co - ordinate')
plt.ylabel('Y Co - ordinate')
plt.grid(True)
plt.show()

fig = category_scatter(x = 'x', y = 'y', label_col = 'dbscan2', data = df5, legend_loc='upper left')
plt.title('DBSCAN with 0.08 eps ')
plt.xlabel('X Co - ordinate')
plt.ylabel('Y Co - ordinate')
plt.grid(True)
plt.show()
