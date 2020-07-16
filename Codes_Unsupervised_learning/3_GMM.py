import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import os, shutil, glob, os.path
import time

start = time.time()

print('Starting Unsupervised Learning (GMM) of Test Images')

# Current Path
cur_path = os.getcwd()

# Output directory
foldername='Unsupervised_GMM_comp35' # Name your folder here
os.makedirs(cur_path + "\\" + foldername)
outdir = os.path.join(cur_path,foldername)

# Intput directory of Test folder
indir = os.path.join(cur_path,'Test')

filelist = glob.glob(os.path.join(indir, '*.png'))
filelist.sort()
c_no = 43 #GMM component number

# Load the extracted feature from the preprocessed data
with open('preprocessed_features.npy', 'rb') as f:
    feature_array = np.load(f)

# Standardizing data (Need to do it before applying PCA Model)
sc = StandardScaler()
sc.fit(feature_array)
feature_array = sc.transform(feature_array)

# Applying PCA Model (PCA Component no. = 50)
pcaModel = PCA(n_components=50)
feature_array = pcaModel.fit_transform(feature_array)

# Applying GMM Model
gmm = GaussianMixture(n_components=c_no, covariance_type='full').fit(feature_array) #random_state=
prediction_gmm = gmm.predict(feature_array)
probs = gmm.predict_proba(feature_array)

# Calculating Silhouette Score
sil = silhouette_score(feature_array,prediction_gmm)
print('Silhouette Score',sil)
print("\n")

# Copying images into their "label" folder
for i, j in enumerate(prediction_gmm):
    try:
        os.makedirs(outdir + "\\" + str(j))
    except OSError:
        pass
    shutil.copy(filelist[i],outdir +"\\"+ str(j)+"\\"+str(i)+".png")

print(f'Time: {(time.time()-start)/60} min')

