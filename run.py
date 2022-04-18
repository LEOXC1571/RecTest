# -*- coding: utf-8 -*-
# @Filename: run
# @Date: 2022-04-17 16:18
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import argparse
import os
import logging
from logging import getLogger
from recbole.config import Config
from recbole.data import data_preparation, save_split_dataloaders
from recbole.utils import init_logger, get_trainer, init_seed, set_color

from model import model_name_map
# from data.dataset import BasedDataset
from recbole.data.dataset import Dataset

def objective_run():
    pass

def run(model=None, dataset=None, saved=False):
    current_path = os.path.dirname(os.path.realpath(__file__))
    # basic config file
    overall_init_file = os.path.join(current_path, 'config/overall.yaml')
    # model config file
    model_init_file = os.path.join(current_path, 'config/model/' + model + '.yaml')
    # dataset config file
    dataset_init_file = os.path.join(current_path, 'config/dataset/' + dataset + '.yaml')

    config_file_list = [overall_init_file, model_init_file, dataset_init_file]
    model_class = model_name_map[model]

    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, action='store', help='model_name')
    parser.add_argument('--dataset', type=str, action='store', help='dataset name')
    parser.add_argument('--saved', str=bool, action='store', help='save result')
    args, unknown = parser.parse_known_args()

    model_name = args.model
    dataset_name = args.dataset
    save_result = args.saved

    run(model_name, dataset_name, save_result)