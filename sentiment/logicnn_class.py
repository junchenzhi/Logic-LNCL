import warnings
import numpy
import torch
import torch.nn as nn
import time
from fol import FOL_But


class LogicNN(object):
    def __init__(self, input=None, network=None, rules=[], C=1.):
        self.input = input
        self.network = network
        self.rules = rules
        self.rule_lambda = [1]
        self.C = C


    def cal_logic(self, y_posterior):
        q_y_given_x = y_posterior
        # combine rule constraints
        distr = self.calc_rule_constraints()
        q_y_given_x *= distr
        # normalize
        n = self.input.shape[0]
        n_q_y_given_x = q_y_given_x / torch.sum(q_y_given_x, 1).reshape((n, 1))
        return n_q_y_given_x


    def calc_rule_constraints(self, new_data=None, new_rule_fea=None):
        if new_rule_fea == None:
            new_rule_fea = [None] * len(self.rules)

        distr_all = torch.zeros([self.input.shape[0], 2]).cuda(0)

        for i, rule in enumerate(self.rules):
            distr = rule.log_distribution(self.C * self.rule_lambda[i], new_data, new_rule_fea[i]).cuda(0)
            distr_all += distr

        distr_y0 = distr_all[:, 0]
        distr_y0 = distr_y0.reshape(distr_y0.shape[0], 1)
        distr_y0_copies = distr_y0.repeat(1, distr_all.shape[1])
        distr_all -= distr_y0_copies

        max = torch.ones([distr_all.shape[0], distr_all.shape[1]]).cuda(0)*60
        min = torch.ones([distr_all.shape[0], distr_all.shape[1]]).cuda(0)*-60

        distr_all = torch.max(torch.min(distr_all, max), min).cuda(0)  # truncate to avoid over-/under-flow
        return torch.exp(distr_all)
