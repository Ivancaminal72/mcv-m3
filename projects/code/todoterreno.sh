#!/bin/bash
#Parameters to test
#Per model
####batch_size = [10, 20, 40, 60, 80, 100]
####epochs = [10, 50, 100]
####optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
####learn_rate = [0.0001 0.001, 0.01, 0.1, 0.2, 0.3]
####momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
#####data augmentation: flip, zoom, rescale, …
#Per layer:
#####activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
#####init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform','he_normal', 'he_uniform'] (Not useful in our case)
#Topology:
#####drop-out layers: p % of inactive weights
#####batchnormalization
#####regularizers
python 4_session.py 32  20 ''Adadelta'' ''relu' 'relu'  'softmax'' 0.2 0.0 test1
