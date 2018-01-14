from __future__ import print_function
import getpass
import os
os.environ["CUDA_VISIBLE_DEVICES"]=getpass.getuser()[-1]

import cPickle
import numpy as np
from PIL import Image, ImageOps
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Reshape

PATCH_SIZE  = 32
PATCH_LEN   = 32
DES_LEN = 1024
NUM_EPOCHS = 150
MLP_DES_DIR = '/home/master03/data/descriptors'+str(PATCH_SIZE)+'_'+str(PATCH_LEN)
PATCHES_DIR = '/home/master03/data/patches'+str(PATCH_SIZE)+'_'+str(PATCH_LEN)
MODEL_FNAME = '/home/master03/data/mlp'+str(PATCH_SIZE)+'_'+str(PATCH_LEN)+'_'+str(NUM_EPOCHS)+'.h5'
classes = {'coast':0,'forest':1,'highway':2,'inside_city':3,'mountain':4,'Opencountry':5,'street':6,'tallbuilding':7}

def build_mlp(input_size=PATCH_SIZE,phase='TRAIN'):
  model = Sequential()
  model.add(Reshape((input_size*input_size*3,),input_shape=(input_size, input_size, 3),name='first'))
  model.add(Dense(units=DES_LEN, activation='relu',name='second'))
  #model.add(Dense(units=1024, activation='relu'))
  if phase=='TEST':
    model.add(Dense(units=8, activation='linear',name='third')) # In test phase we softmax the average output over the image patches
  else:
    model.add(Dense(units=8, activation='softmax',name='third'))
  return model

#Find maximum number of patches for one image
maxPatch = 0
zeroPatchNumber = False
for imname in os.listdir(os.path.join(PATCHES_DIR+'/train', 'tallbuilding')):
    if maxPatch < int(imname.split('_')[1].split('.')[0]):
        maxPatch = int(imname.split('_')[1].split('.')[0])
    if not zeroPatchNumber:
        if int(imname.split('_')[1].split('.')[0]) is 0:
            zeroPatchNumber = True

#Define architecture of the model and load the weigths
model = build_mlp(input_size=PATCH_SIZE)

#Load computed weights
model.load_weights(MODEL_FNAME)

#Get the output of a hidden layer
sec_layer = Model(inputs=model.input, outputs=model.get_layer('second').output)

#Load Train patches and compute descriptors
D_train = []
L_train = []
for class_dir in os.listdir(PATCHES_DIR+'/train'):
    cls = classes[class_dir]
    for imname in os.listdir(os.path.join(PATCHES_DIR+'/train', class_dir)):
        patch = Image.open(os.path.join(PATCHES_DIR+'/train', class_dir, imname))
        patch = np.expand_dims(np.array(patch), axis=0)
        des = sec_layer.predict(patch/ 255.)
        D_train.append(np.array(des))
        L_train.append(cls)

#Load Test patches and compute descriptors
D_test = []
L_test = []
for class_dir in os.listdir(PATCHES_DIR+'/test'):
    cls = classes[class_dir]
    for imname in os.listdir(os.path.join(PATCHES_DIR+'/test',class_dir)):
      patch = Image.open(os.path.join(PATCHES_DIR+'/test',class_dir,imname))
      patch = np.expand_dims(np.array(patch), axis=0)
      out = model.predict(patch/255.)
      D_test.append(np.array(sec_layer.predict(patch / 255.0)))
      L_test.append(cls)

#Save to memory
print("Saving MLP_2048 labels & descriptors...")
if not os.path.exists(MLP_DES_DIR+ '/test') or not os.path.exists(MLP_DES_DIR+ '/train'):
    os.makedirs(MLP_DES_DIR + '/test')
    os.makedirs(MLP_DES_DIR + '/train')
cPickle.dump(D_train, open(MLP_DES_DIR + '/train/MLP_' + str(DES_LEN) + '_descriptors.dat', "wb"))
cPickle.dump(L_train, open(MLP_DES_DIR + '/train/MLP_' + str(DES_LEN) + '_labels.dat', "wb"))
cPickle.dump(D_test, open(MLP_DES_DIR + '/test/MLP_' + str(DES_LEN) + '_descriptors.dat', "wb"))
cPickle.dump(L_test, open(MLP_DES_DIR + '/test/MLP_' + str(DES_LEN) + '_labels.dat', "wb"))


