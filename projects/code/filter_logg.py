import numpy as np
import sys

loss = []
accu = []
test = []

save_dir = sys.argv[1] #./my_directory/
with open(save_dir + "log.txt", "r") as logger:
    for line in logger:
        lossaccu = line[0:41].split(' ')
        loss.append(float(lossaccu[0].translate(None, ' ][,')))
        accu.append(float(lossaccu[1].translate(None, ' ][,')))

    loss_mu = np.mean(np.array(loss))
    accu_mu = np.mean(np.array(accu))

    logger.seek(0)
    filter_logger = open(save_dir + "filter.txt", "w")
    for i, line in enumerate(logger):
        print i
        if loss[i] < loss_mu and accu[i] > accu_mu:
            filter_logger.write(line)
    filter_logger.close()
    logger.close()