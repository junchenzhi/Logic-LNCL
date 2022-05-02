"""

First Order Logic (FOL) rules

"""

import warnings
import numpy
import torch



class FOL_But(object):
    def __init__(self, K, input, fea):
        self.input = input
        self.fea = fea
        self.K = K


    def log_distribution(self, w, X=None, F=None):
        if F == None:
            X, F = self.input, self.fea
        F_mask = F[:, 0]
        F_fea = F[:, 1:]
        # y = 0
        distr_y0 = w * F_mask * F_fea[:, 0]
        # y = 1
        distr_y1 = w * F_mask * F_fea[:, 1]

        distr_y0 = distr_y0.reshape([distr_y0.shape[0], 1])
        distr_y1 = distr_y1.reshape([distr_y1.shape[0], 1])

        distr = torch.cat((distr_y0, distr_y1), 1)
        return distr