#4_session.py batch_size epochs optimizer act1,act2,act3 learn_rate
#momentum data_augmentation init_mode drop_out image_name

import os
import getpass
os.environ["CUDA_VISIBLE_DEVICES"]=getpass.getuser()[-1]

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Convolution2D
from keras import backend as K
from keras.callbacks import Callback
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import warnings
import time
import sys

init = time.time()

train_data_dir='/share/datasets/MIT_split/train'
val_data_dir='/share/datasets/MIT_split/test'
test_data_dir='/share/datasets/MIT_split/test'

#Arguments
batch_size         = int(sys.argv[1]) #32
epoch_num          = int(sys.argv[2]) #20
optimizer          = sys.argv[3] #'Adadelta'
activation_layers  = sys.argv[4].split(',') #relu,relu,softmax
learn_rate         = float(sys.argv[5]) #2.0
momentum           = float(sys.argv[6]) #2.0
dropout            = bool (int(sys.argv[7])) #True False
save_dir            = sys.argv[8] #./my_directory/
rotation_range      = 0 #6 #int (sys.argv[8])
width_shift_range   = 0 #0.2#float(sys.argv[9])
height_shift_range  = 0 #0.2#float(sys.argv[10])
shear_range         = 0 #0#float(sys.argv[11])
zoom_range          = 0 #0.4#float(sys.argv[12])
channel_shift_range = 0 #0#float(sys.argv[13])
horizontal_flip     = 0 #bool(int(sys.argv[14]))
vertical_flip       = 0 #0#bool(int(sys.argv[15]))
lastEpoch = 0
dir_data = './../data/s4/'

filename = str(batch_size) + ' ' + str(epoch_num) + ' ' + str(optimizer) + ' ' + str(activation_layers) + ' ' + \
           str(learn_rate) + ' ' + str(momentum) + ' ' + str(dropout) + ' ' + str(rotation_range) + ' ' + str(width_shift_range) \
           + ' ' + str(height_shift_range) + ' ' + str(shear_range) + ' ' + str(zoom_range) + ' ' + str(channel_shift_range) + ' ' + str(horizontal_flip) \
           + ' ' + str(vertical_flip)

print('\n'+'\n'+'\n'+filename+'\n'+'\n'+'\n')

img_width = 224
img_height=224

if len(activation_layers) < 3:
  raise AssertionError()

#Set a maximum number of epoch (stop with the callback)
set_epoch_max = False
if set_epoch_max:
    epoch_num = 50

if dropout:
    epoch_num = 80
    warnings.warn("Warning: epoch_max is set to :" + str(epoch_num) + " and epoch number parameter is not used")

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

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.001, verbose=1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value   = value
        self.verbose = verbose
    def on_epoch_end(self, epoch, logs={}):
        global lastEpoch
        current = logs.get("val_loss")
        if current != None and current < self.value:
            self.model.stop_training = True
            lastEpoch = epoch + 1

class EarlyStoppingByAccVal(Callback):
    def __init__(self, monitor='val_acc', value=0.25, verbose=1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
    def on_epoch_end(self, epoch, logs={}):
        global lastEpoch
        acu_train = logs.get("acc")
        acu_val = logs.get("val_acc")
        if acu_train != None and acu_val != None and (acu_val-acu_train) > self.value:
            self.model.stop_training = True
            lastEpoch = epoch + 1
    
# create the base pre-trained model
base_model = VGG16(weights='imagenet')
plot_model(base_model, to_file='modelVGG16a.png', show_shapes=True, show_layer_names=True)

x = base_model.layers[-9].output
x = Convolution2D(512, 3, 3, activation='relu')(x)
x = GlobalAveragePooling2D()(x)
#x = Flatten()(x)
x = Dense(units=4096, activation=activation_layers[0],name='firstfull')(x)
if dropout:
    x = Dropout(0.5)(x)
x = Dense(units=1024, activation=activation_layers[1],name='secondfull')(x)
if dropout:
    x = Dropout(0.5)(x)
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
  optimizer = optimizers.Adam(lr=learn_rate)
elif optimizer == 'Adamax':
  optimizer = optimizers.Adamax(lr=learn_rate)
elif optimizer == 'Nadam':
  optimizer = optimizers.Nadam(lr=learn_rate)

model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
for layer in model.layers:
    print layer.name, layer.trainable

datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=True,
    featurewise_std_normalization=False,
    samplewise_std_normalization=True,
    preprocessing_function=preprocess_input,
    rotation_range=rotation_range,
    width_shift_range=width_shift_range,
    height_shift_range=height_shift_range,
    shear_range=shear_range,
    zoom_range=zoom_range,
    channel_shift_range=channel_shift_range,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=horizontal_flip,
    vertical_flip=vertical_flip,
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
        samples_per_epoch=np.floor(400/batch_size)*batch_size,
        nb_epoch=epoch_num,
        validation_data=validation_generator,
        validation_steps=807//batch_size,
        callbacks = [EarlyStoppingByLossVal(), EarlyStoppingByAccVal()])

result = model.evaluate_generator(test_generator, val_samples=807//batch_size)
print model.metrics_names
print result

if not os.path.exists(dir_data):
    os.makedirs(dir_data)
model.save_weights(dir_data+filename+'.h5')

#Second training
model.load_weights(dir_data+filename+'.h5')
for layer in base_model.layers:
    layer.trainable = True
model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])

for layer in model.layers:
    print layer.name, layer.trainable

history2=model.fit_generator(
        train_generator,
        samples_per_epoch=np.floor(400/batch_size)*batch_size,
        nb_epoch=epoch_num,
        validation_data=validation_generator,
        validation_steps=807//batch_size,
        callbacks = [EarlyStoppingByLossVal(), EarlyStoppingByAccVal()])


# list all data in history
if True:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # results
    logger = open(save_dir + "log.txt", "a")
    logger.write(str(result) + '     ' + filename+'\n')
    logger.close()

    # summarize history for accuracy
    plt.plot(history2.history['acc'])
    plt.plot(history2.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_dir +'accuracy' + filename + '.jpg')
    plt.close()
    # summarize history for loss
    plt.plot(history2.history['loss'])
    plt.plot(history2.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_dir +'loss' + filename + '.jpg')


end = time.time()
print 'Done in ' + str(end - init) + ' secs.\n'