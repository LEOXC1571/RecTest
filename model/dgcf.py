# -*- coding: utf-8 -*-
# @Filename: dgcf
# @Date: 2022-04-21 08:42
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import random as rd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

class DGCF(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def  __init__(self, config, dataset):
        super(DGCF, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.n_factores = config['n_factors']
        self.n_iterations = config['n_iterations']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.cor_weight = config['cor_weight']
    pass
