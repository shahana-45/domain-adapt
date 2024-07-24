import argparse
import logging as lgg
import time
from importlib import import_module

from run import Config
from train_mutual_learning import test
from ldsgm_utils import build_iterator, build_dataset, get_time_dif

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LDSGM', help='choose a model')
    parser.add_argument('--cuda', type=int, default=0, choices=[0, 1], help='choose a cuda: 0 or 1')
    parser.add_argument('--tune', type=int, default=0, choices=[1, 0], help='fine tune or not: 0 or 1')
    parser.add_argument('--base', type=str, default='roberta', choices=['roberta'], help='roberta model as encoder')
    parser.add_argument('--lambda4', type=float, default=1.0, help='lambda for kl loss')
    args = parser.parse_args()

    dataset = 'PDTB/novel'  # data (test folder contains DiscoGem dataset)
    config = Config(dataset, args.cuda, bool(args.tune), args.base, args.lambda4)

    # load trained model
    model_name = args.model  # bert
    x = import_module(model_name)
    model = x.Model(config).to(config.device)

    config = Config(dataset, args.cuda, bool(args.tune), args.base, args.lambda4)

    start_time = time.time()
    lgg.info("Loading data...")

    train_data, dev_data, test_data = build_dataset(config, True)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    lgg.info("Time usage: {}".format(time_dif))

    #dev accuracy TODO get scores on dev set
    #acc_top_dev, f1_top_dev, acc_sec_dev, f1_sec_dev, acc_conn_dev, f1_conn_dev \
    #   = test(config, model, dev_iter, only_eval=True)

    # test accuracy
    acc_top_test, f1_top_test, acc_sec_test, f1_sec_test, acc_conn_test, f1_conn_test \
        = test(config, model, test_iter, only_eval=True)
