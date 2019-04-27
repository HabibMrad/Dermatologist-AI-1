from sklearn.datasets import load_files   
from keras.utils import np_utils  
import numpy as np
from glob import glob
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input
from tqdm import tqdm 
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2

# Transfere Learning section
# preprocess_input method is imported above from keras.applications.inception_resnet_v2
train_imgs_preprocess = preprocess_input(train_tensors)
valid_imgs_preprocess = preprocess_input(valid_tensors)
test_imgs_preprocess = preprocess_input(test_tensors)
del train_tensors, valid_tensors, test_tensors


transfer_model = InceptionResNetV2(include_top=False)

train_data = transfer_model.predict(train_imgs_preprocess)
valid_data = transfer_model.predict(valid_imgs_preprocess)
test_data = transfer_model.predict(test_imgs_preprocess)

del train_imgs_preprocess, valid_imgs_preprocess, test_imgs_preprocess

