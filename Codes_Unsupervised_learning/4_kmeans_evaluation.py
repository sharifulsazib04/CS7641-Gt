import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
import glob, os.path
import time

start = time.time()

print('Starting Elbow Method varying # of clusters (KMeans)')

# Load the extracted feature from the preprocessed data
with open('preprocessed_features.npy', 'rb') as f:
    feature_array = np.load(f)

# Standardizing data (Need to do it before applying PCA Model)
sc = StandardScaler()
sc.fit(feature_array)
feature_array = sc.transform(feature_array)

# Applying PCA Model (Choose PCA Component no. = 50 to get a better and efficient clustering)
pcaModel = PCA(n_components=50)
feature_array = pcaModel.fit_transform(feature_array)

model=KMeans()
KK=np.arange(5,420,45)

# To visualize distortion score vs # of clusters
visualizer = KElbowVisualizer(model, k=KK, locate_elbow=False)
visualizer.fit(feature_array)
visualizer.show()

# To visualize silhouette score vs # of clusters
visualizer_s = KElbowVisualizer(model, k=KK, metric = 'silhouette', locate_elbow=False, timings=False)
visualizer_s.fit(feature_array)
visualizer_s.show()

print(f'Time: {(time.time()-start)/60} min')


