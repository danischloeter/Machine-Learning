# %% --------------------------------------- Imports -------------------------------------------------------------------
from keras.models import load_model
import os
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
from imblearn.over_sampling import RandomOverSampler
from keras.initializers import glorot_uniform
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import cv2
from keras.layers import Activation, Dropout, Flatten, Dense
# This script generates the training set (the one you will be provided with),
# and the held out set (the one we will use to test your model towards the leaderboard).


# Dataset = "train"

#unzip the files and saving them to a folder train
with zipfile.ZipFile("train.zip","r") as z:
    z.extractall(".")

train_data_dir = "train/"
pngcounter = len(glob.glob1(train_data_dir, "*.png"))
print("Count of images",pngcounter)
txtcounter = len(glob.glob1(train_data_dir, "*.txt"))
print("Count of tags",txtcounter)
imgs = glob.glob1(train_data_dir, "*.png")
tags= glob.glob1(train_data_dir, "*.txt")


Alltargets=[]
xtag=[]
unknown=0
for i in range(len(imgs)):
    # for j in range(len(tags)):
        #if imgs[i].split('.')[0]==tags[j].split('.')[0]:
            path=pjoin('train/',imgs[i].split('.')[0]+'.txt')
            # path2=pjoin(path,'.txt')
            file1 = open(path, "r")
            classe=file1.read()
            xtag.append(str(imgs[i]))
            Alltargets.append(classe)

#Now lets convert our categorical data
le = LabelEncoder()
Alltargets = le.fit_transform(Alltargets)
print(le.classes_)
print(Alltargets)

# countredbloodcell=print(len( os.listdir('/home/ubuntu/redbloodcell/') ))
# countring=print(len( os.listdir('/home/ubuntu/ring/') ))
# countscvhizont=print(len( os.listdir('/home/ubuntu/schizont/') ))
# counttrophozoite=print(len( os.listdir('/home/ubuntu/trophozoite/') ))


# Now lets divide our data into train and test (with test_size=0.1 and random_state=0)
# Shuffle the data
# xtag, Alltargets=shuffle(xtag,Alltargets,random_state=0)
print(Alltargets)

x_train, x_test, y_train, y_test = train_test_split(xtag, Alltargets, test_size=0.1, random_state=0, stratify=Alltargets)

np.save("/home/ubuntu/Exam 1/x_train1.npy", x_train)
#print('x_train',x_train)
#print(np.unique(x_train))
np.save("/home/ubuntu/Exam 1/y_train1.npy", y_train)
# print('y_train',y_train)
plt.hist(Alltargets)
plt.show()
print(np.unique(y_train))
np.save("/home/ubuntu/Exam 1/x_test1.npy", x_test)
#print('x_test',x_test)
#print(np.unique(x_test))
np.save("/home/ubuntu/Exam 1/y_test1.npy", y_test)
print('y_test',y_test)
print(np.unique(y_test))


# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_uniform(seed=SEED)
# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 0.001
N_NEURONS = 400
N_NEURONS2 =200
N_NEURONS3 = 100
N_EPOCHS = 100
BATCH_SIZE = 512
DROPOUT = 0.2
# %% -------------------------------------- Data Prep ------------------------------------------------------------------

x, y = np.load("/home/ubuntu/Exam 1/x_train1.npy"), np.load("/home/ubuntu/Exam 1/y_train1.npy")


# Allimages = np.zeros((len(x),100,100,3))
# #size=[]
#
# for i,g in zip(x,range(len(x))):
# # load the image
#     img = cv2.imread(pjoin('/home/ubuntu/train/', i), cv2.COLOR_BGR2GRAY)
#     img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_CUBIC)
#     Allimages[g] = img

x_train, x_testv, y_train, y_testv = train_test_split(x, y, random_state=SEED, test_size=0.2,stratify=y,shuffle=True)
x_train, x_testv = x_train.reshape(len(x_train), -1), x_testv.reshape(len(x_testv), -1)
x_train, x_test = x_train/255, x_testv/255
y_train, y_testv = to_categorical(y_train, num_classes=4), to_categorical(y_testv, num_classes=4)


# datagen = ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')


print('shape x_train',x_train.shape)
print('shape y_train',y_train.shape)
print('shape x_testv',x_testv.shape)


# %% -------------------------------------- Training Prep ----------------------------------------------------------

model = Sequential([
    #Flatten(),
    Dense(N_NEURONS2, input_dim=30000, activation="relu"),
    Dense(N_NEURONS2, input_dim=30000, activation="relu"),
    Dense(N_NEURONS2, input_dim=30000, activation="relu"),
    Dense(N_NEURONS, input_dim=30000, activation="relu"),
    Dense(N_NEURONS, input_dim=30000, activation="relu"),
    Dense(N_NEURONS, input_dim=30000, activation="relu"),
    Dense(N_NEURONS3, input_dim=30000, activation="relu"),

    Dense(4, activation="softmax")
])

model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

# # compute quantities required for featurewise normalization
# # (std, mean, and principal components if ZCA whitening is applied)
# datagen.fit(x_train)

#%% -------------------------------------- Training Loop ----------------------------------------------------------

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_testv, y_testv),
         callbacks=[ModelCheckpoint("/home/ubuntu/Exam 1/mlp_dschloeter.hdf5", monitor="val_loss", save_best_only=True)])

# fits the model on batches with real-time data augmentation:
# model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE), validation_data=(x_testv, y_testv),
#           callbacks=[ModelCheckpoint("/home/ubuntu/Exam 1/mlp_dschloetertest.hdf5", monitor="val_loss", save_best_only=True)],
#                     steps_per_epoch=len(x_train) / BATCH_SIZE, epochs=N_EPOCHS)
# %% ------------------------------------------ Final test -------------------------------------------------------------
modelbest = load_model('/home/ubuntu/Exam 1/mlp_dschloeter.hdf5')
print("Final accuracy on validations set:", 100*modelbest.evaluate(x_testv, y_testv)[1], "%")
print("Cohen Kappa", cohen_kappa_score(np.argmax(modelbest.predict(x_testv),axis=1),np.argmax(y_testv,axis=1)))
print("F1 score", f1_score(np.argmax(modelbest.predict(x_testv),axis=1),np.argmax(y_testv,axis=1), average = 'macro'))
