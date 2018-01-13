from __future__ import print_function
import os
import getpass
os.environ["CUDA_VISIBLE_DEVICES"]=getpass.getuser()[-1]

#user defined variables
PATCH_SIZE  = 64
PATCHES_DIR = '../data/MIT_split_patches'
DESCRIPTORS_DIR = '../data/MIT_split_descriptors'
MODEL_FNAME = '/home/master03/ivan/patch_based_mlp.h5'

import cPickle
from keras.models import Model
from patch_based_mlp_MIT_8_scene import build_mlp
from sklearn.feature_extraction import image
from PIL import Image
classes = {'coast':0,'forest':1,'highway':2,'inside_city':3,'mountain':4,'Opencountry':5,'street':6,'tallbuilding':7}
correct = 0.
total   = 807
count   = 0

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
        D_train.append(np.array(sec_layer.predict(patch / 255.0)))
        L_train.append(cls)

#Load Test patches and compute descriptors
D_test = []
L_test = []
for class_dir in os.listdir(PATCHES_DIR+'/test'):
    cls = classes[class_dir]
    for imname in os.listdir(os.path.join(PATCHES_DIR+'/test',class_dir)):
      im = Image.open(os.path.join(PATCHES_DIR+'/test',class_dir,imname))
      patches = image.extract_patches_2d(np.array(im), (PATCH_SIZE, PATCH_SIZE), max_patches=1.0)
      out = model.predict(patches/255.)
      for patch in patches:
          D_test.append(np.array(sec_layer.predict(patch / 255.0)))
          L_test.append(cls)

#Save to memory
print("Saving MLP_2048 labels & descriptors...")
cPickle.dump(D_train, open(os.path.join(DESCRIPTORS_DIR,'/train','MLP_2048_descriptors.dat'), "wb"))
cPickle.dump(L_train, open(os.path.join(DESCRIPTORS_DIR,'/train','MLP_2048_labels.dat'), "wb"))
cPickle.dump(D_test, open(os.path.join(DESCRIPTORS_DIR,'/test','MLP_2048_descriptors.dat'), "wb"))
cPickle.dump(L_test, open(os.path.join(DESCRIPTORS_DIR,'/test','MLP_2048_labels.dat'), "wb"))


