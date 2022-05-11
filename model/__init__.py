# -*- coding: utf-8 -*-
# @Filename: __init__.py
# @Date: 2022-04-17 16:18
# @Author: Leo Xu
# @Email: leoxc1571@163.com

from recbole.model.general_recommender import Pop, BPR
from .lightgcn import LightGCN
from .dgcf_ref import DGCF
from .ngcf import NGCF
from .sgl import SGL
from .simgcl import SimGCL

recoble_models = {
    'BPR',
    'Pop'
}

model_name_map = {
    'Pop': Pop,
    'BPR': BPR,
    'LGCN': LightGCN,
    'NGCF': NGCF,
    'DGCF': DGCF,
    'SGL': SGL,
    'SimGCL': SimGCL
}