# -*- coding: utf-8 -*-
# @Filename: __init__.py
# @Date: 2022-04-17 16:18
# @Author: Leo Xu
# @Email: leoxc1571@163.com

from recbole.model.general_recommender import Pop, BPR

recoble_models = {
    'BPR',
    'Pop'
}

model_name_map = {
    'Pop': Pop,
    'BPR': BPR
}