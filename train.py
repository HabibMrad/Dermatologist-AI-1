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
from keras.layers import Conv2D, Dropout, Flatten, Dense, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

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


# Adding trainable layer to the imported arch.
my_model = Sequential()

# add 2 fully connected layer to the end of the arch. and train it on our dataset
my_model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))
my_model.add(Dropout(0.4)) # to eliminate overfitting
my_model.add(Dense(1024, activation='relu'))
my_model.add(Dropout(0.4))



# last layer should be 3 perceptrons only because we have only 3 classes in our dataset
my_model.add(Dense(3, activation='softmax'))

print(my_model.summary())

# compile our model
my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


checkpoint_filepath = 'weights.hdf5'
my_checkpointer = ModelCheckpoint(filepath=checkpoint_filepath,
                               verbose=1, save_best_only=True)

# now our model is ready for training .. lets do that
my_model.fit(train_data, train_targets, 
          validation_data=(valid_data, valid_targets),
          epochs=60, batch_size=200, callbacks=[my_checkpointer], verbose=1)


