import numpy as np
import os, glob, os.path
from skimage import exposure
from skimage import transform
from skimage import exposure
from skimage import io
import time

start = time.time()

print('Starting Feature Extraction of Test Images')

#Intput directory of "Test" folder
cur_path = os.getcwd()
path = os.path.join(cur_path,'img_comp')
images = os.listdir(path)

# Image cropping dimensions
l1 = 4
l2 = 28

featurelist = []

# Loop over images, preprocess them and extract features

for elem in images:
    try:
        image = io.imread(path + '\\'+ elem)
        image = transform.resize(image,(32,32)) #Resizing all images to 32 by 32
        image = exposure.equalize_adapthist(image, clip_limit=0.1) #CLAHE: to enhance the contrast of an image
        image = image[l1:l2, l1:l2]
        featurelist.append(image.flatten())

        print('Successfully extracted features of image no.', elem)
    except:
        print('Error extracting features of image no.',elem)


feature_array = np.array(featurelist)

with open('preprocessed_features.npy', 'wb') as f:
    np.save(f, feature_array)

print(f'Time: {(time.time()-start)/60} min')