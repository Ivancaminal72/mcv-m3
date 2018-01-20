#!/bin/bash
#Parameters to test

#Per model
batch_size=("10" "20" "40" "60" "80" "100")
epochs=('10' '20' '50' '100')
optimizer=('SGD' 'RMSprop' 'Adagrad' 'Adadelta' 'Adam' 'Adamax' 'Nadam')
learn_rate=('0.0001' '0.001' '0.01' '0.1' '0.2' '0.3')
momentum=('0.0' '0.2' '0.4' '0.6' '0.8' '0.9')
#####data augmentation: flip, zoom, rescale, …

#Per layer:
activation=('softplus' 'softsign' 'relu' 'tanh' 'sigmoid' 'hard_sigmoid' 'linear') #softmax out considered only for last layer
#####init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform','he_normal', 'he_uniform'] (Not useful in our case)

#Topology:
#####drop-out layers: p % of inactive weights
#####batchnormalization
#####regularizers

while true
do
    r1=$(shuf -i 0-5 -n 1)
    #epoch not shufed
    r3=$(shuf -i 0-6 -n 1)
    r4=$(shuf -i 0-5 -n 1)
    r5=$(shuf -i 0-5 -n 1)
    r6_1=$(shuf -i 0-6 -n 1)
    r6_2=$(shuf -i 0-6 -n 1)
    python 4_session.py ${batch_size[r1]} 20 ${optimizer[r3]} ${activation[r6_1]},${activation[r6_2]},softmax ${learn_rate[r4]} ${momentum[r5]} ./dale_gas/
done