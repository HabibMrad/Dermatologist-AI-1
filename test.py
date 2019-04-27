from sklearn.datasets import load_files   
from keras.utils import np_utils  
import numpy as np
from glob import glob
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input
from tqdm import tqdm 
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

