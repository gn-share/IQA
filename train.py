import random
import numpy as np
from BaseIQASolver import BaseIQASolver
from args import Configs


def main(config):
    print = config.printf

    sel_num = config.img_num[config.dataset]

    srcc_all = np.zeros(config.train_test_num, dtype=np.float64)
    plcc_all = np.zeros(config.train_test_num, dtype=np.float64)

    print('Training and testing on %s dataset for %d rounds...' %
          (config.dataset, config.train_test_num))
    if config.random_single:
        random.shuffle(sel_num)
    for i in range(config.train_test_num):
        print('Round %d' % (i+1))
        if not config.random_single:
            random.shuffle(sel_num)
        train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
        test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]

        solver = BaseIQASolver(
            config, config.folder_path[config.dataset], train_index, test_index)
        srcc_all[i], plcc_all[i] = solver.train()

    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)

    print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' %
          (srcc_med, plcc_med))
    print(srcc_all)
    print(plcc_all)


if __name__ == '__main__':

    main(Configs())
