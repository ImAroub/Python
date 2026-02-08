#Group members:
#Aroub Alfehaid   431201578
#Hoor Alrobah     432216220
#Rasha Alsamaani  431201586
#Leen Suliman     431201407
#Lama Aldokail    431201598

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Dataset description:
# This dataset contains geometric measurements of wheat seeds.
# The features describe seed shape and size, making it suitable for clustering.

# Step 1:Load dataset, assign column names, drop Class column(because it's not needed for clustering) and select features
file_path = "seeds_dataset.txt"
df = pd.read_csv(file_path, sep=r'\s+', header=None)

df.columns = ['Area', 'Perimeter', 'Compactness', 'Length', 'Width', 'Asymmetry', 'Groove Length', 'Class']
df = df.drop(columns=['Class'])

features = df[['Area', 'Perimeter', 'Compactness', 'Length', 'Width', 'Asymmetry', 'Groove Length']]


# Step 2: Normalize the data using StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# Step 3: Apply K-Means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0)
df['Cluster'] = kmeans.fit_predict(features_scaled)


# step 4: Use PCA for dimensionality reduction to visualize clusters in 2D
pca = PCA(n_components=2)
pca_features = pca.fit_transform(features_scaled)


# step 5: Plot the 2D projection of the clusters
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=df['Cluster'], cmap='viridis')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Seeds Dataset Clusters (K-Means)')
plt.show()


# Step 6: Display DataFrame with clusters and calculate mean of each feature per cluster
print(df) #prints all data points after assigning each one to a cluster.
cluster_means = df.groupby('Cluster').mean()
print("\nMean of each feature for each Cluster:")
print(cluster_means) #prints only 3 rows showing mean value of each feature for each cluster
# Cluster interpretation:
# We chose K=3 because the dataset originally contains three wheat varieties.
# Cluster 0 represents seeds with smaller area and perimeter, Cluster 1 represents medium-sized seeds,
# and Cluster 2 represents larger seeds with higher compactness and groove length, showing clear geometric differences.
