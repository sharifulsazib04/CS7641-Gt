import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import glob, os.path
import matplotlib.pyplot as plt
import time

# Class calculating distortion score
class Distortion(object):
    def __init__(self):
        pass

    def pairwise_dist(self,x, y):
        np.random.seed(1)
        x_norm = np.sum(x ** 2, axis=1, keepdims=True)
        yt = y.T
        y_norm = np.sum(yt ** 2, axis=0, keepdims=True)
        dist2 = np.abs(x_norm + y_norm - 2.0 * np.dot(x, yt))
        return np.sqrt(dist2)

    def _get_loss(self, centers, cluster_idx, points):
        K = centers.shape[0]
        dist_center = self.pairwise_dist(points, centers)
        xpi = cluster_idx[:, np.newaxis] #cluster_idx is a N by 1 vector: values of the the vector are center numbers, indices of the vector are point number
        zerox = np.zeros((points.shape[0],K))
        row, col = np.indices(xpi.shape)

        zerox[row,xpi] = dist_center[row,xpi] ** 2
        loss = np.sum(zerox)
        return loss

start = time.time()

print('Starting Evaluation of KMeans Model varying # of PCA components')

# Load the extracted feature from the preprocessed data
with open('preprocessed_features.npy', 'rb') as f:
    feature_array = np.load(f)

# Standardizing data (Need to do it before applying PCA Model)
sc = StandardScaler()
sc.fit(feature_array)
feature_array = sc.transform(feature_array)

sil = []
DB=[]
dist = []
KK=43 # Cluster Number
pca_comp=np.arange(50,1000,200)
Dis = Distortion()

# Loop over number of PCA components
for p in pca_comp:
    print('PCA component Number: ',p)
    # Applying PCA Model
    pcaModel = PCA(n_components = p)
    feature_pca = pcaModel.fit_transform(feature_array)
    # Applying KMeans Model
    kmeans = KMeans(init='k-means++', n_clusters=KK, random_state=0).fit(feature_pca)
    kmeans.fit(feature_pca)
    cluster_labels = kmeans.fit_predict(feature_pca)

    # Calculating Distortion, Silhouette score and Davies Bouldin Index

    dist.append(Dis._get_loss(kmeans.cluster_centers_, cluster_labels, feature_pca))
    sil.append(silhouette_score(feature_pca, cluster_labels))
    DB.append(davies_bouldin_score(feature_pca, cluster_labels))


# Plotting distortion scores vs # of PCA components
plt.figure(1)
plt.plot(pca_comp, dist, 'bo-')
plt.xlabel('# of Clusters, k')
plt.ylabel('Distortion Score')
plt.show()

# Plotting Silhouette scores and Davies Bouldin indices vs # of Clusters
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('# of Clusters, k')
ax1.set_ylabel('Silhouette Score', color=color)
ax1.plot(pca_comp, sil, 'bx-')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('Davies Bouldin Index', color=color)
ax2.plot(pca_comp, DB, 'ro-')
ax2.tick_params(axis='y', labelcolor=color)
plt.show()

print(f'Time: {(time.time()-start)/60} min')
