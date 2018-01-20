#!/bin/bash
#Parameters to test

#Per model
r1_bs=("10" "20" "40" "60" "80" "100")
r2_opt=('SGD' 'Adadelta' 'Adam' 'Adamax') #RMSprop Adagrad (need more than 20 epoch); Nadam (irregular)
r3_lr=('0.000001' '0.00001' '0.0001' '0.001' '0.01') # '0.1' '0.2' '0.3' (bad results)
r4_mom=('0.9' '1.2' '1.5' '1.8' '2.1' '2.4') # '0.0' '0.2' '0.4' '0.6' '0.8' (bad results)
#####data augmentation: flip, zoom, rescale, …

#Per layer:
r5_act=('softsign' 'relu' 'tanh' 'hard_sigmoid' 'linear') #softplus  sigmoid (bad) softmax (last)
#####init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform','he_normal', 'he_uniform'] (Not useful in our case)

#Topology:
#####drop-out layers: p % of inactive weights
#####batchnormalization
#####regularizers

while true
do
    r1=$(shuf -i 0-5 -n 1)
    r2=$(shuf -i 0-3 -n 1)
    r3=$(shuf -i 0-5 -n 1)
    r4=$(shuf -i 0-5 -n 1)
    r5_1=$(shuf -i 0-4 -n 1)
    r5_2=$(shuf -i 0-4 -n 1)
    python 4_session.py ${r1_bs[r1]} ${r2_opt[r2]} ${r5_act[r5_1]},${r5_act[r5_2]},softmax ${r3_lr[r3]} ${r4_mom[r4]} ./dale_gas/
done