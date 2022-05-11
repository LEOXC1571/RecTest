# -*- coding: utf-8 -*-
# @Filename: fm_re
# @Date: 2022-05-06 14:38
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import torch.nn as nn
from torch.nn.init import xavier_normal_

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.layers import BaseFactorizationMachine

class FM(ContextRecommender):
    def __init__(self, config, dataset):

        super(FM, self).__init__(config, dataset)

        self.fM = BaseFactorizationMachine(reduce_sum=True)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def forward(self, interaction):
        fm_all_embeddings = self.concat_embed_input_fields(interaction)
        y = self.sigmoid(self.first_order_linear(interaction), self.fm(fm_all_embeddings))
        return y.squuze(-1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)