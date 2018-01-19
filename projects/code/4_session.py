#Arguments 
#batch_size
#epochs
#optimizer
#activation_layers
#learn_rate
#momentum
#data_augmentation
#init_mode
#drop_out
#image_name

import os
import getpass
os.environ["CUDA_VISIBLE_DEVICES"]=getpass.getuser()[-1]

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import sys

"""first_arg = sys.argv[1]
second_arg = sys.argv[2]
print len(sys.argv)
print first_arg
print second_arg"""

train_data_dir='/share/datasets/MIT_split/train'
val_data_dir='/share/datasets/MIT_split/test'
test_data_dir='/share/datasets/MIT_split/test'

batch_size        = int(sys.argv[1]) #32
number_of_epoch   = int(sys.argv[2]) #20
optimizer         = sys.argv[3] #'Adadelta'
activation_layers = sys.argv[4].split()
learn_rate        = float(sys.argv[5])
momentum          = float(sys.argv[6])
image_name        = sys.argv[7]

img_width = 224
img_height=224

if( len(activation_layers) < 3 ):
  raise AssertionError()


def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        # 'RGB'->'BGR'
        x = x[ ::-1, :, :]
        # Zero-center by mean pixel
        x[ 0, :, :] -= 103.939
        x[ 1, :, :] -= 116.779
        x[ 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    return x
    
# create the base pre-trained model
base_model = VGG16(weights='imagenet')
plot_model(base_model, to_file='modelVGG16a.png', show_shapes=True, show_layer_names=True)

x = base_model.layers[-9].output
x = GlobalAveragePooling2D()(x)
#x = Flatten()(x)
x = Dense(units=4096, activation=activation_layers[0],name='firstfull')(x)
x = Dense(units=4096, activation=activation_layers[1],name='secondfull')(x)
x = Dense(8, activation=activation_layers[2],name='predictions')(x)
model = Model(input=base_model.input, output=x)

plot_model(model, to_file='modelVGG16b.png', show_shapes=True, show_layer_names=True)
for layer in base_model.layers:
    layer.trainable = False

print (optimizer)
if optimizer == 'Adadelta':
  optimizer = optimizers.Adadelta(lr=learn_rate)
elif optimizer == 'SGD':
  optimizer = optimizers.SGD(lr=learn_rate,momentum=momentum)
elif optimizer == 'Adam':
  optimizer = optimizers.Adam(lr=learn_rate,momentum=momentum)
elif optimizer == 'Adamax':
  optimizer = optimizers.Adamax(lr=learn_rate,momentum=momentum)
elif optimizer == 'Nadam':
  optimizer = optimizers.Nadam(lr=learn_rate,momentum=momentum)
model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
for layer in model.layers:
    print layer.name, layer.trainable

#preprocessing_function=preprocess_input,
datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=True,
    featurewise_std_normalization=False,
    samplewise_std_normalization=True,
	preprocessing_function=preprocess_input,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None)

train_generator = datagen.flow_from_directory(train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

test_generator = datagen.flow_from_directory(test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = datagen.flow_from_directory(val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

history=model.fit_generator(
        train_generator,
        samples_per_epoch=416, #13 batches
        nb_epoch=number_of_epoch,
        validation_data=validation_generator,
        validation_steps=807//batch_size)

result = model.evaluate_generator(test_generator, val_samples=807//batch_size)
print model.metrics_names
print result


# list all data in history

if True:
  # summarize history for accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig(image_name+'accuracy.jpg')
  plt.close()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig(image_name+'loss.jpg')