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
train_tensors = paths_to_tensor(tqdm(train_files))
valid_tensors = paths_to_tensor(tqdm(valid_files))
test_tensors = paths_to_tensor(tqdm(test_files))

# Data Augmentation
apply_train_image_transform = False 
# I Can't do data augmentation on colab because the RAM crashed after using all variable 
# and you will need to run the previous cells again(take long time ) 
if apply_train_image_transform:
    # Caution: Doesn't guarantee prevention of duplication.
    datagen_train = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True)
    
    datagen_train.fit(train_tensors)
    shape = (train_tensors.shape[0] * 2,) + train_tensors.shape[1:]
    generated = np.ndarray(shape=shape)
    for i, image in tqdm(enumerate(train_tensors)):
        generated[i] = datagen_train.random_transform(image)
    
    train_tensors = np.concatenate((train_tensors, generated))
    train_targets = train_targets.repeat(2, axis=0)