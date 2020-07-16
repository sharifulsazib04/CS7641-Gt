import numpy as np
from sklearn import mixture
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import os, shutil, glob, os.path
import matplotlib.pyplot as plt
import time

start = time.time()

print('Starting evaluation of GMM model varying # of PCA components')

with open('preprocessed_features.npy', 'rb') as f:
    feature_array = np.load(f)

# Standardizing data (Need to do it before applying PCA Model)
sc = StandardScaler()
sc.fit(feature_array)
feature_array = sc.transform(feature_array)

bic_score = []
bar = []
sil = []
DB = []
pca_comp = np.arange(50, 1000, 200)
gmm_comp = 35
cov_type = 'full'

# Looping over # of PCA components

for p in pca_comp:
    print('PCA Component Number: ', p)
    # Applying PCA Model
    pcaModel = PCA(n_components = p)
    feature_pca = pcaModel.fit_transform(feature_array)
    # Fitting a GMM Model with EM Algorithm
    gmm = mixture.GaussianMixture(n_components=gmm_comp, covariance_type=cov_type)
    gmm.fit(feature_pca)
    prediction_gmm = gmm.predict(feature_pca)

    bic_score.append(gmm.bic(feature_pca))
    sil.append(silhouette_score(feature_pca, prediction_gmm))
    DB.append(davies_bouldin_score(feature_pca, prediction_gmm))

bic_score = np.array(bic_score)
pca_comp = np.array(pca_comp)
bar_width = 160

# Plotting BIC scores vs # of components
plt.figure(figsize=(6, 4))

plt.xticks(pca_comp)
plt.bar(pca_comp, bic_score, width=bar_width)
plt.ylim([bic_score.min() * 1.03 - bic_score.max() * 0.03, bic_score.max()+1e3])
plt.xlabel('Number of GMM components')
plt.ylabel('BIC Score')
plt.show()

# Plotting Silhouette scores and Davies Bouldin indices vs # of PCA components
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('# of PCA Components')
ax1.set_ylabel('Silhouette Score', color=color)
ax1.plot(pca_comp, sil, 'bx-')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('Davies Bouldin Index', color=color)
ax2.plot(pca_comp, DB, 'ro-')
ax2.tick_params(axis='y', labelcolor=color)
plt.show()

print(f'Time: {(time.time() - start) / 60} min')
