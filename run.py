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
from data.dataset import TagBasedDataset
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
    #config init
    config = Config(model=model_class, dataset=dataset, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    # logger init
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    # dataset = TagBasedDataset(config)
    dataset = Dataset(config)
    if config['save_dataset']:
        dataset.save()
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    if config['save_dataloaders']:
        save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

    # model loading and init
    model = model_class(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and init
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    logger.info(set_color('best valid', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, action='store', help='model_name')
    parser.add_argument('--dataset', type=str, action='store', help='dataset name')
    parser.add_argument('--save', action='store_true', help='save result', default=False)
    args, unknown = parser.parse_known_args()

    model_name = args.model
    dataset_name = args.dataset
    save_result = args.save

    run(model_name, dataset_name, save_result)