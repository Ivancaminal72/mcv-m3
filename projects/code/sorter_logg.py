import numpy as np
import sys

loss = []
accu = []
lines = []
filter_lines = []

save_dir = sys.argv[1] #./my_directory/
with open(save_dir + "log.txt", "r") as logger:
    for line in logger:
        lines.append(line)
        lossaccu = line[0:41].split(' ')
        loss.append(float(lossaccu[0].translate(None, ' ][,')))
        accu.append(float(lossaccu[1].translate(None, ' ][,')))

    Loss = np.array(loss)
    Accu = np.array(accu)
    loss_max = np.max(Loss)
    accu_max = np.max(Accu)
    iLoss = np.argsort(Loss)
    iAccu = np.argsort(-Accu)

    mix = []
    for i in range(len(loss)):
        mix.append(((Accu[i])/accu_max) - (Loss[i])/loss_max)

    pLoss = np.zeros((iLoss.shape))
    pAccu = np.zeros((iLoss.shape))
    for i in range(len(iLoss)):
        pLoss[iLoss[i]] = i
        pAccu[iAccu[i]] = i

    iMix = np.argsort(-np.array(mix))
    sorter_logger = open(save_dir + "sorter.txt", "w")
    for i, idx in enumerate(iMix):
        print i
        sorter_logger.write('LOSS ' + str(pLoss[idx])+ ' ACCU ' + str(pAccu[idx]) + ' ' + lines[idx])
    sorter_logger.close()
    logger.close()