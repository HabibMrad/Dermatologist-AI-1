"""
Created on Sat Apr 27 19:10:40 2019

@author: Gabir N. Yusuf
"""
from sklearn.datasets import load_files   
from keras.utils import np_utils  
import numpy as np
from glob import glob
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input
from tqdm import tqdm 
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


def load_dataset(data_path, shuffle=None):
    kwargs = {}
    if shuffle != None:
        kwargs['shuffle'] = shuffle
    data = load_files(data_path, **kwargs) # scikit-learn func imported above to load the data
    img_files = np.array(data['filenames'])
    
    #one-hot encoding for dataset labels
    targets = np_utils.to_categorical(np.array(data['target']), 3)
    return img_files, targets


def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(299, 299)) #image sizes is arch based
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


def paths_to_tensor(image_paths):
    return np.vstack([path_to_tensor(path) for path in image_paths])



train_files, train_targets = load_dataset('train')
valid_files, valid_targets = load_dataset('valid')
test_files, test_targets = load_dataset('test', shuffle=False)

# load lables
label_name = [item[6:-1] for item in sorted(glob("train/*/"))]

# Loading images in tensors
print('\nLoading training files into train_tensors ... ')
train_tensors = paths_to_tensor(tqdm(train_files))
print('\nLoading Validation files into  valid_tensors ... ')
valid_tensors = paths_to_tensor(tqdm(valid_files))
print('\nLoading Test files into test_tensors ... ')
test_tensors = paths_to_tensor(tqdm(test_files))

