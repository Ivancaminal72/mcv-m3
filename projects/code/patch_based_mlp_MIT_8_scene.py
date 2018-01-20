from __future__ import print_function
import os
import getpass
os.environ["CUDA_VISIBLE_DEVICES"]=getpass.getuser()[-1]

from utils import *
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator

#user defined variables
PATCH_SIZE  = 128
PATCH_LEN   = 8
DES_LEN = 2048
NUM_EPOCHS= 5
BATCH_SIZE  = 16
DATASET_DIR = '/share/datasets/MIT_split/'
#PATCHES_DIR = '/home/master03/data/patches'+str(PATCH_SIZE)+'_'+str(PATCH_LEN)
PATCHES_DIR = '/share/datasets/MIT_split/'
MODEL_FNAME = '/home/master03/data/mlp'+str(PATCH_SIZE)+'_'+str(PATCH_LEN)+'_'+str(NUM_EPOCHS)+'_'+str(DES_LEN)+'.h5'


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

if not os.path.exists(DATASET_DIR):
  print('ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
  quit()
if not os.path.exists(PATCHES_DIR):
  print('WARNING: patches dataset directory '+PATCHES_DIR+' do not exists!\n')
  print('Creating image patches dataset into '+PATCHES_DIR+'\n')
  generate_image_patches_db(DATASET_DIR,PATCHES_DIR,PATCH_SIZE,PATCH_LEN)
  print('Done!\n')


print('Building MLP model...\n')

model = build_mlp(input_size=PATCH_SIZE)

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

print(model.summary())

print('Done!\n')

if not os.path.exists(MODEL_FNAME):
    print('WARNING: model file '+MODEL_FNAME+' do not exists!\n')
    print('Start training...\n')
    # this is the dataset configuration we will use for training
    # only rescaling
    train_datagen = ImageDataGenerator(
          rescale=1./255,
          horizontal_flip=True)
    # this is the dataset configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)
    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
          PATCHES_DIR+'/train',  # this is the target directory
          target_size=(PATCH_SIZE, PATCH_SIZE),  # all images will be resized to PATCH_SIZExPATCH_SIZE
          batch_size=BATCH_SIZE,
          classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
          class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical L_train
    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
          PATCHES_DIR+'/test',
          target_size=(PATCH_SIZE, PATCH_SIZE),
          batch_size=BATCH_SIZE,
          classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
          class_mode='categorical')
    history = model.fit_generator(
          train_generator,
          steps_per_epoch=18810 // BATCH_SIZE,
          epochs=NUM_EPOCHS,
          validation_data=validation_generator,
          validation_steps=8070 // BATCH_SIZE)
    print('Done!\n')
    print('Saving the model into '+MODEL_FNAME+' \n')
    model.save_weights(MODEL_FNAME)  # always save your weights after training or during training
    print('Done!\n')

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('accuracy.jpg')
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('loss.jpg')


print('Building MLP model for testing...\n')
model = build_mlp(input_size=PATCH_SIZE, phase='TEST')
print(model.summary())

print('Done!\n')

print('Loading weights from '+MODEL_FNAME+' ...\n')
print ('\n')

model.load_weights(MODEL_FNAME)

print('Done!\n')

print('Start evaluation ...\n')

directory = DATASET_DIR+'/test'
classes = {'coast':0,'forest':1,'highway':2,'inside_city':3,'mountain':4,'Opencountry':5,'street':6,'tallbuilding':7}
correct = 0.
total   = 807
count   = 0

for class_dir in os.listdir(directory):
    cls = classes[class_dir]
    for imname in os.listdir(os.path.join(directory,class_dir)):
      im = Image.open(os.path.join(directory,class_dir,imname))
      patches = image.extract_patches_2d(np.array(im), (PATCH_SIZE, PATCH_SIZE), max_patches=PATCH_LEN)
      out = model.predict(patches/255.)
      predicted_cls = np.argmax( softmax(np.mean(out,axis=0)) )
      if predicted_cls == cls:
        correct+=1
      count += 1
      print('Evaluated images: '+str(count)+' / '+str(total)+'\n')
    
print('Done!\n')
print('Test Acc. = '+str(correct/total)+'\n')
