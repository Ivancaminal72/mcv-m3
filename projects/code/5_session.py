import os
import getpass
os.environ["CUDA_VISIBLE_DEVICES"]=getpass.getuser()[-1]

from keras.applications.imagenet_utils import _obtain_input_shape as _obtain_input_shape
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D,Flatten, Input, MaxPooling2D
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

#python 5_session.py 32 20 'Adadelta' relu,relu,softmax 0.0001 2.0 0 ./mini_vgg/
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
dir_data = './../data/s5/'

filename = str(batch_size) + ' ' + str(epoch_num) + ' ' + str(optimizer) + ' ' + str(activation_layers) + ' ' + \
           str(learn_rate) + ' ' + str(momentum) + ' ' + str(dropout) + ' ' + str(rotation_range) + ' ' + str(width_shift_range) \
           + ' ' + str(height_shift_range) + ' ' + str(shear_range) + ' ' + str(zoom_range) + ' ' + str(channel_shift_range) + ' ' + str(horizontal_flip) \
           + ' ' + str(vertical_flip)

if not os.path.exists(dir_data):
    os.makedirs(dir_data)

print('\n'+'\n'+'\n'+filename+'\n'+'\n'+'\n')

img_width = 224
img_height=224

if len(activation_layers) < 3:
  raise AssertionError()

#Set a maximum number of epoch (stop with the callback)
set_epoch_max = False
if set_epoch_max:
    epoch_num = 50

#Change epochs if dropout
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
    def __init__(self, monitor='loss', value=0.001, verbose=1):
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
    def __init__(self, monitor='val_acc', value=0.2, verbose=1):
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
    
# create model from scratch
input_shape = _obtain_input_shape(None, 224, 48, K.image_data_format(),True)
img_input = Input(shape=input_shape)

# Block 1
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(x)

# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv3')(x)
x = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(x)

# Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = MaxPooling2D(pool_size=(2, 2), name='block3_pool')(x)

# Block 4
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
x = MaxPooling2D(pool_size=(4, 4), name='block4_pool')(x)

#x = Conv2D(24, (3, 3), activation='relu')(x)
#x = GlobalAveragePooling2D()(x)
# Block 5
#x = Conv2D(24, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
#x = Conv2D(24, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
#x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block5_pool')(x)
x = Flatten(name='flatten')(x)
x = Dense(units=2048, activation=activation_layers[0],name='firstfull')(x)
if dropout:
    x = Dropout(0.5)(x)
x = Dense(units=512, activation=activation_layers[1],name='secondfull')(x)
if dropout:
    x = Dropout(0.5)(x)
x = Dense(8, activation=activation_layers[2],name='predictions')(x)
model = Model(input=img_input, output=x)

plot_model(model, to_file=dir_data+filename+'.png', show_shapes=True, show_layer_names=True)

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
    preprocessing_function=preprocess_input, #TODO: check this!!
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

#TODO: Recheck this in project_feedback.pdf
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
        steps_per_epoch=1881 // batch_size, #np.floor(400/batch_size)*batch_size
        nb_epoch=epoch_num,
        validation_data=validation_generator,
        validation_steps=807//batch_size,
        callbacks = [EarlyStoppingByLossVal(), EarlyStoppingByAccVal()])

result = model.evaluate_generator(test_generator, val_samples=807//batch_size)
print model.metrics_names
print result

#model.save_weights(dir_data+filename+'.h5')

# list all data in history
if True:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # results
    logger = open(save_dir + "log.txt", "a")
    logger.write(str(result) + '     ' + filename+'\n')
    logger.close()

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_dir +'accuracy' + filename + '.jpg')
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_dir +'loss' + filename + '.jpg')


end = time.time()
print 'Done in ' + str(end - init) + ' secs.\n'