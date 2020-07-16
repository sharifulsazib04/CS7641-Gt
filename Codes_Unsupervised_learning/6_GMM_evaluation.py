import numpy as np
import itertools
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

print('Starting evaluation of GMM model varying # of GMM components')

with open('preprocessed_features.npy', 'rb') as f:
    feature_array = np.load(f)

# Standardizing data (Need to do it before applying PCA Model)
sc = StandardScaler()
sc.fit(feature_array)
feature_array = sc.transform(feature_array)

# Applying PCA Model
pcaModel = PCA(n_components=50)
feature_array = pcaModel.fit_transform(feature_array)

lowest_score = np.infty
bic_score = []
sil = []
DB = []
gmm_comp = np.arange(5,70,10)
cov_type = ['spherical', 'tied', 'diag', 'full']

# Looping over # of GMM components

for c in cov_type:
    for n in gmm_comp:
        print('GMM Component Number: ',n)
        # Fitting a GMM Model with EM Algorithm
        gmm = mixture.GaussianMixture(n_components=n, covariance_type=c)
        gmm.fit(feature_array)
        prediction_gmm = gmm.predict(feature_array)

        bic_score.append(gmm.bic(feature_array))
        sil.append(silhouette_score(feature_array,prediction_gmm))
        DB.append(davies_bouldin_score(feature_array,prediction_gmm))
        if bic_score[-1] < lowest_score:
            lowest_score = bic_score[-1]
            bestGMM = gmm

bic_score = np.array(bic_score)
sil = np.array(sil)
DB = np.array(DB)
color_i = itertools.cycle(['red', 'green', 'orange', 'blue'])
bar_width = 2

# Plotting BIC scores vs # of components
plt.figure(figsize=(10, 8))
b_bar = []
for i, (c, color) in enumerate(zip(cov_type, color_i)):
    x_pos = np.array(gmm_comp) + bar_width * (i - 2)
    b_bar.append(plt.bar(x_pos, bic_score[i * len(gmm_comp):(i + 1) * len(gmm_comp)],width=bar_width, color=color))

plt.xticks(gmm_comp)
plt.ylim([bic_score.min() * 1.01 - .01 * bic_score.max(), bic_score.max()])
plt.title('BIC score for different covariance types')
plt.xlabel('Number of GMM components')
plt.ylabel('BIC Score')
plt.legend([b[0] for b in b_bar], cov_type)
plt.show()

plt.figure(figsize=(10, 8))
s_bar = []
for i, (c, color) in enumerate(zip(cov_type, color_i)):
    x_pos = np.array(gmm_comp) + bar_width * (i - 2)
    s_bar.append(plt.bar(x_pos, sil[i * len(gmm_comp):(i + 1) * len(gmm_comp)],width=bar_width, color=color))

plt.xticks(gmm_comp)
plt.ylim([sil.min() * 1.01 - .01 * sil.max(), sil.max()+0.02])
plt.title('Silhouette score for different covariance types')
plt.xlabel('Number of GMM components')
plt.ylabel('Silhouette Score')
plt.legend([b[0] for b in s_bar], cov_type)
plt.show()

plt.figure(figsize=(10, 8))
db_bar = []
for i, (c, color) in enumerate(zip(cov_type, color_i)):
    x_pos = np.array(gmm_comp) + bar_width * (i - 2)
    db_bar.append(plt.bar(x_pos, DB[i * len(gmm_comp):(i + 1) * len(gmm_comp)],width=bar_width, color=color))

plt.xticks(gmm_comp)
plt.ylim([DB.min() * 1.01 - .01 * DB.max(), DB.max()+0.05])
plt.title('Davies Bouldin Index for different covariance types')
plt.xlabel('Number of GMM components')
plt.ylabel('Davies Bouldin Index')
plt.legend([b[0] for b in db_bar], cov_type)
plt.show()

print(f'Time: {(time.time()-start)/60} min')
