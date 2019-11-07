# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
os.system("sudo pip install imblearn")
from keras.models import load_model
import random
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.metrics import cohen_kappa_score, f1_score
import numpy as np
import zipfile
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import glob
from os.path import join as pjoin
import shutil
from sklearn.preprocessing import LabelEncoder
### OJO!!!
from imblearn.over_sampling import RandomOverSampler
from keras.initializers import glorot_uniform
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import cv2
from keras.layers import Activation, Dropout, Flatten, Dense



def predict(x):
    # Here x is a NumPy array. On the actual exam it will be a list of paths.
    # %% --------------------------------------------- Data Prep -------------------------------------------------------
    Allimages = np.zeros((len(x),100,100,3))
    size = []
    for i, g in zip(x, range(len(x))):
        # load the image
        # img = load_img(pjoin('/home/ubuntu/redbloodcell/',i),color_mode = "grayscale",target_size=(100,100))
        img = cv2.imread(i,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_CUBIC)
        # print(img.size)
        # size.append(img.size)
        # convert to numpy array
        # img_array = img_to_array(img)
        # Allimages[g]=img_array
        Allimages[g] = img

    x = Allimages.reshape(len(x), -1)
    x = x / 255

    # Write any data prep you used during training
    # %% --------------------------------------------- Predict ---------------------------------------------------------
    model = load_model('mlp_dschloeter.hdf5')
    # If using more than one model to get y_pred, they need to be named as "mlp_ajafari1.hdf5", ""mlp_ajafari2.hdf5", etc.
    y_pred = np.argmax(model.predict(x), axis=1)

    return y_pred, model
    # If using more than one model to get y_pred, do the following:
    # return y_pred, model1, model2  # If you used two models
    # return y_pred, model1, model2, model3  # If you used three models, etc.

