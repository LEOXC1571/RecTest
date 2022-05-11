# -*- coding: utf-8 -*-
# @Filename: SimGCL
# @Date: 2022-05-10 14:14
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import numpy as np
import scipy.sparse as sp
import torch

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
import torch.nn.functional as F

class SimGCL(GeneralRecommender):

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SimGCL, self).__init__(config, dataset)
        self._user = dataset.inter_feat[dataset.uid_field]
        self._item = dataset.inter_feat[dataset.iid_field]
        # load config
        self.emb_size = config['embedding_size']
        self.cl_rate = config['ssl_weight']
        self.reg_weight = config['reg_weight']
        self.eps = config['eps']
        self.n_layers = config['n_layers']
        # emb generation
        self.user_embedding = torch.nn.Embedding(self.n_users, self.emb_size)
        self.item_embedding = torch.nn.Embedding(self.n_items, self.emb_size)
        self.train_graph = self.csr2tensor(self.create_adjust_matrix())
        self.reg_loss = EmbLoss()
        self.restore_user_e = None
        self.restore_item_e = None
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def get_ego_embeddings(self):
        _user_embeddings = self.user_embedding.weight
        _item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([_user_embeddings, _item_embeddings], dim=0)
        return ego_embeddings

    def graph_construction(self):
        self.graph = self.csr2tensor(self.create_adjust_matrix())

    def create_adjust_matrix(self):
        matrix = None
        ratings = np.ones_like(self._user, dtype=np.float32)
        matrix = sp.csr_matrix((ratings, (self._user, self._item + self.n_users)),
                               shape=(self.n_users + self.n_items, self.n_users + self.n_items))
        matrix = matrix + matrix.T
        D = np.array(matrix.sum(axis=1)) + 1e-7
        D = np.power(D, -0.5).flatten()
        D = sp.diags(D)
        return D.dot(matrix).dot(D)

    def csr2tensor(self, matrix: sp.csr_matrix):
        matrix = matrix.tocoo()
        x = torch.sparse.FloatTensor(
            torch.LongTensor(np.array([matrix.row, matrix.col])),
            torch.FloatTensor(matrix.data.astype(np.float32)), matrix.shape
        ).to(self.device)
        return x

    def forward(self, graph, perturbed=True):
        main_ego = self.get_ego_embeddings()
        all_ego = [main_ego]

        for i in range(self.n_layers):
            main_ego = torch.sparse.mm(graph, main_ego)
            if perturbed:
                random_state = torch.rand(main_ego.shape)
                main_ego += torch.mul(torch.sign(main_ego), F.normalize(random_state, p=2, dim=1)) * self.eps
            all_ego.append(main_ego)
        all_ego = torch.stack(all_ego, dim=1)
        all_ego = torch.mean(all_ego, dim=1, keepdim=False)
        user_emb, item_emb = torch.split(all_ego, [self.n_users, self.n_items], dim=0)
        return user_emb, item_emb

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user_list = interaction[self.USER_ID]
        pos_item_list = interaction[self.ITEM_ID]
        neg_item_list = interaction[self.NEG_ITEM_ID]
        user_emb, item_emb = self.forward(self.train_graph, perturbed=False)
        user_p1, item_p1 = self.forward(self.graph, perturbed=True)
        user_p2, item_p2 = self.forward(self.graph, perturbed=True)

        total_loss = self.calc_bpr_loss(user_emb, item_emb, user_list, pos_item_list, neg_item_list) + \
                     self.calc_ssl_loss(user_list, pos_item_list, user_p1, user_p2, item_p1, item_p2)

        return total_loss
    def calc_bpr_loss(self, user_emb, item_emb, user_list, pos_item_list, neg_item_list):
        u_e = user_emb[user_list]
        pi_e = item_emb[pos_item_list]
        ni_e = item_emb[neg_item_list]
        p_scores = torch.mul(u_e, pi_e).sum(dim=1)
        n_scores = torch.mul(u_e, ni_e).sum(dim=1)

        l1 = torch.sum(-F.logsigmoid(p_scores - n_scores))

        u_e_p = self.user_embedding(user_list)
        pi_e_p = self.item_embedding(pos_item_list)
        ni_e_p = self.item_embedding(neg_item_list)

        l2 = self.reg_loss(u_e_p, pi_e_p, ni_e_p)

        return l1 + l2 * self.reg_weight

    def calc_ssl_loss(self, user_list, pos_item_list, user_1, user_2, item_1, item_2):
        u_emb1 = F.normalize(user_1[user_list], dim=1)
        u_emb2 = F.normalize(user_2[user_list], dim=1)
        i_emb1 = F.normalize(item_1[pos_item_list], dim=1)
        i_emb2 = F.normalize(item_2[pos_item_list], dim=1)

        v1 = torch.sum(u_emb1 * u_emb2, dim=1)
        v2 = u_emb1.matmul(u_emb2.T)
        v1 = torch.exp(v1 / 0.2)
        v2 = torch.sum(torch.exp(v2 / 0.2), dim=1)
        ssl_user = -torch.sum(torch.log(v1 / v2))

        v3 = torch.sum(i_emb1 * i_emb2, dim=1)
        v4 = i_emb1.matmul(i_emb2.T)
        v3 = torch.exp(v3 / 0.2)
        v4 = torch.sum(torch.exp(v4 / 0.2), dim=1)
        ssl_item = -torch.sum(torch.log(v3 / v4))

        return (ssl_user + ssl_item) * self.cl_rate

    def predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward(self.train_graph)

        user = self.restore_user_e[interaction[self.USER_ID]]
        item = self.restore_item_e[interaction[self.ITEM_ID]]
        return torch.sum(user * item, dim=1)

    def full_sort_predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward(self.train_graph)

        user = self.restore_user_e[interaction[self.USER_ID]]
        return user.matmul(self.restore_item_e.T)

    def train(self, mode: bool=True):
        T = super().train(mode=mode)
        if mode:
            self.graph_construction()
        return T