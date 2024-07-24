import argparse
import os
import random
from importlib import import_module
import jsonlines

import numpy as np
import pyprind
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging as lgg
from sklearn import metrics
import time
from datetime import datetime

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig

import training
from DomainAdaptation import metrics
from DomainAdaptation.DAConfig import DAConfig
from DomainAdaptation.confidence_scores import class4, class11
from DomainAdaptation.model import get_dev_scores, annotate
from GOLFConfig import GOLFConfig
from golf_utils import MyDataset
from run import Config
from ldsgm_utils import build_dataset, build_iterator, get_time_dif
from train_mutual_learning import evaluate
from utils.utils import utils
from pytorch_pretrained_bert.optimization import BertAdam


def add_model_id_to_path(path, model_id):
    """
    Parameters
    ----------
    path: str
    model_id: int

    Returns
    -------
    str
    """
    return "%s.%d" % (path, model_id)


def train(
        da_config,
        idrr_model_list,
        labeled_data,
        unlabeled_dataset,
        dev_iter,
        test_iter,
        path_train_losses,
        path_snapshot,
        path_ann,
        path_dev_pred,
        path_dev_gold,
        path_dev_eval,
        golf_args=None):
    n_models = len(idrr_model_list)
    # PDTB3 training set
    if da_config.model == "LDSGM":
        labeled_dataset = build_iterator(labeled_data, idrr_model_list[0].config)
    elif da_config.model == "GOLF":
        labeled_dataset = DataLoader(dataset=labeled_data,
                                     batch_size=da_config.batch_size,
                                     shuffle=True)
    print(len(labeled_dataset))
    print(len(labeled_data))

    # Get optimizers and schedulers
    n_labeled = len(labeled_data)
    max_epoch = da_config.max_epoch

    batch_size = da_config.batch_size
    if da_config.bootstrapping_type == "so":
        total_update_steps = n_labeled * max_epoch // batch_size
        warmup_steps = int(total_update_steps * da_config.warmup_ratio)
    elif da_config.bootstrapping_type in ["st", "ct"]:
        # NOTE: In bootstrapping, `total_update_steps` is not the actual total number of update steps,
        #       because the number of selected pseudo-labeled_ldsgm data is the same with or smaller than
        #       the number of sampled unlabeled data (=`unlabeled_data_sampling_size`).
        total_update_steps = (n_labeled + da_config.unlabeled_data_sampling_size) * max_epoch // batch_size
        warmup_steps = 7000
    elif da_config.bootstrapping_type in ["tt", "at"]:
        total_update_steps = (n_labeled + da_config.unlabeled_data_sampling_size * 2) * max_epoch // batch_size
        warmup_steps = 7000
    else:
        raise Exception("Never occur.")

    utils.writelog("*********************Training*********************")
    utils.writelog("n_labeled: %d" % n_labeled)
    utils.writelog("max_epoch: %d" % max_epoch)
    utils.writelog("batch_size: %d" % batch_size)
    utils.writelog("total_update_steps: %d" % total_update_steps)
    utils.writelog("warmup_steps: %d" % warmup_steps)

    writer_train = jsonlines.Writer(open(path_train_losses, "w"), flush=True)
    writer_dev = jsonlines.Writer(open(path_dev_eval, "a"), flush=True)
    bestscore_holders = {"joint": utils.BestScoreHolder(scale=1.0)}
    bestscore_holders["joint"].init()
    bestscore_holders["independent"] = [None for _ in range(n_models)]
    for p_i in range(n_models):
        bestscore_holders["independent"][p_i] = utils.BestScoreHolder(scale=1.0)
        bestscore_holders["independent"][p_i].init()

    ##################
    # Initial validation phase
    ##################

    best_f1 = -1.0
    with torch.no_grad():
        for p_i in range(n_models):
            # do LDSGM/GOLF validation only
            if da_config.model == "LDSGM":
                loss_test, acc_top_test, f1_top_test, acc_sec_test, f1_sec_test, acc_conn_test, f1_conn_test \
                    = get_dev_scores("LDSGM", dev_iter, idrr_model_list[0], da_config,
                                     add_model_id_to_path(path_dev_pred, p_i))
            elif da_config.model == "GOLF":
                loss_test, acc_top_test, f1_top_test, acc_sec_test, f1_sec_test, acc_conn_test, f1_conn_test \
                    = get_dev_scores("GOLF", dev_iter, idrr_model_list[0], golf_args,
                                     add_model_id_to_path(path_dev_pred, p_i))

            # create metric score for model
            scores = metrics.attachment_scores(
                pred_path=add_model_id_to_path(path_dev_pred, p_i),
                gold_path=path_dev_gold, f1=f1_sec_test, accuracy=acc_sec_test)
            scores["f1"] *= 100.0
            # print(scores["f1"])
            scores["epoch"] = 0
            writer_dev.write(scores)
            utils.writelog(utils.pretty_format_dict(scores))

            bestscore_holders["independent"][p_i].compare_scores(scores["f1"], 0)

            # Save the model
            # idrr_model_list[0].save_model(path=add_model_id_to_path(path_snapshot, p_i))
            torch.save(idrr_model_list[p_i].state_dict(), add_model_id_to_path(path_snapshot, p_i))

            if (da_config.bootstrapping_type != "at") or (da_config.bootstrapping_type == "at" and p_i == 0):
                if best_f1 < scores["f1"]:
                    best_f1 = scores["f1"]
                print(best_f1)
    bestscore_holders["joint"].compare_scores(best_f1, 0)

    ##################
    # /Initial validation phase
    ##################

    ##################
    # Training-and-validation loops
    ##################
    y = import_module(da_config.model)  # new model for self-training
    # new LDSGM model
    if da_config.model == "LDSGM":
        semisupervised_model = y.Model(ldsgm_config).to(ldsgm_config.device)  # model to train using semi-supervision
    elif da_config.model == "GOLF":  # TODO change to GOLF
        semisupervised_model = y.Model(golf_args).to(ldsgm_config.device)  # model to train using semi-supervision

    ldsgm_params = {
        "total_batch": 0,
        "dev_best_acc_top": 0.0,
        "dev_best_acc_sec": 0.0,
        "dev_best_acc_conn": 0.0,
        "dev_best_f1_top": 0.0,
        "dev_best_f1_sec": 0.0,
        "dev_best_f1_conn": 0.0,
    }

    # Optimizers
    # For GOLF, only the 'optimizer' variable is used
    # LDSGM uses both 'optimizer' and 'optimizer_reverse' variables
    param_optimizer = list(semisupervised_model.named_parameters())
    #param_optimizer = list(idrr_model_list[0].named_parameters())
    optimizer = None
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters_rev = [
        {'params': [p for n, p in param_optimizer if any(rd in n for rd in ['decoder_reverse'])]}]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    if da_config.model == "LDSGM":
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=ldsgm_config.learning_rate,
                             warmup=0.05,
                             t_total=total_update_steps)

    optimizer_reverse = BertAdam(optimizer_grouped_parameters_rev,
                                 lr=ldsgm_config.learning_rate,
                                 warmup=0.05,
                                 t_total=total_update_steps)

    new_model = idrr_model_list[0]  # initially annotate with source-only model
    seed = 42
    for epoch in range(1, max_epoch + 1):
        ##################
        # Annotation phase
        ##################
        # call LDSGM evaluate() to annotate unlabeled dataset
        if da_config.bootstrapping_type == "so":
            # In the source-only system, the training dataset consists of only manually labeled_ldsgm data
            if da_config.model == "LDSGM":
                train_dataset_list = [labeled_dataset]


        elif (epoch == 1) or ((epoch - 1) % da_config.annotation_reflesh_frequency == 0):
            # 0. Reflesh the training dataset
            train_dataset_list = []
            # 1. Sample unlabeled documents' indices
            utils.writelog("1. Sampling unlabeled documents ...")

            # print(len(unlabeled_dataset))
            np.random.seed(seed)
            subset_indices = np.random.permutation(len(unlabeled_dataset))[
                             :da_config.unlabeled_data_sampling_size]
            seed = seed + 1

            if da_config.model == "LDSGM":
                unlabeled_iter = build_iterator(np.array(unlabeled_dataset)[subset_indices], new_model.config)
            elif da_config.model == "GOLF":
                unlabeled_iter = build_iterator(np.array(unlabeled_dataset)[subset_indices], ldsgm_config)

            # 2. Annotate the sampled unlabeled documents and measure the confidence scores
            utils.writelog("2. Annotating sampled unlabeled documents and measuring the confidence scores ...")

            path_ann_list = [add_model_id_to_path(path_ann, p_i) for p_i in range(n_models)]
            with torch.no_grad():
                for p_i in range(n_models):
                    ann_list_top, ann_list_sec, _, confidence_scores = annotate("GOLF", unlabeled_iter, new_model,
                                                                                da_config,
                                                                                add_model_id_to_path(path_dev_pred,
                                                                                                     p_i))

            # 3. Select pseudo-labeled_ldsgm data using a sample selection criterion
            utils.writelog("3. Selecting pseudo-labeled_ldsgm data ...")
            if da_config.bootstrapping_type == "st":
                pseudo_labeled_dataset_list, info \
                    = select_pseudo_labeled_data_for_selftraining(
                    unlabeled_dataset=np.array(unlabeled_dataset)[subset_indices],
                    ann_list_top=ann_list_top,
                    ann_list_sec=ann_list_sec,
                    confidence_scores_list=confidence_scores,
                    topk_ratio=da_config.topk_ratio)

            # TODO add co-training bootstrapping method

            # Print info.
            for p_i in range(n_models):
                n_pseudo_labeled = len(pseudo_labeled_dataset_list[p_i])
                ratio = float(n_pseudo_labeled) / len(subset_indices) * 100.0
                utils.writelog(
                    "[Epoch %d; Model %d] Number of pseudo-labeled_ldsgm data: %d (Utility: %.02f%%; Range: [%.02f, %.02f])" % \
                    (epoch, p_i, n_pseudo_labeled, ratio, info[p_i]["min_score"], info[p_i]["max_score"]))

            # 4. Combine the labeled_ldsgm and pseudo-labeled_ldsgm datasets
            utils.writelog("4. Combining the labeled_ldsgm and pseudo-labeled_ldsgm datasets ...")
            for p_i in range(n_models):
                # pseudo_labeled_data = np.asarray(pseudo_labeled_dataset_list[p_i], dtype="O")
                pseudo_labeled_data = pseudo_labeled_dataset_list[p_i]
                # print(pseudo_labeled_data)
                if da_config.bootstrapping_type == "at" and p_i == 0:
                    # In asymmetric tri-training, we do not use the labeled_ldsgm source dataset for training the target-domain model (p_i=0)
                    train_dataset = pseudo_labeled_data
                else:
                    print(len(labeled_data))
                    print(len(pseudo_labeled_data))
                    train_dataset = np.concatenate(
                        [np.array(labeled_data.content)[:, :12], np.array(pseudo_labeled_data)],
                        axis=0)  # combine labeled and unlabeled data (PDTB3 + genre-specific unlabeled data)
                train_dataset_list.append(train_dataset)

        ##################
        # /Annotation phase
        ##################

        ##################
        # Training phase
        ##################
        # call LDSGM/GOLF train
        for p_i in range(n_models):

            train_dataset = train_dataset_list[p_i]
            n_train = len(train_dataset)
            perm = np.random.permutation(n_train)
            train_dataset = train_dataset[perm]
            # print(train_dataset)
            #######################
            # LDSGM/GOLF training
            #######################

            if da_config.model == "LDSGM":
                new_train_iter = build_iterator(train_dataset, ldsgm_config)
                utils.writelog("Beginning LDSGM training with pseudo-labels.....")

                # train LDSGM model
                ldsgm_params["optimizer"] = optimizer
                ldsgm_params["optimizer_reverse"] = optimizer_reverse

                train_ldsgm(ldsgm_config, semisupervised_model, new_train_iter, dev_iter, test_iter, ldsgm_params)

            elif da_config.model == "GOLF":
                train_dataset = MyDataset(args, args.data_file + 'nofile.txt', dataset=train_dataset,
                                          load_content=False)
                # GOLF data loader for labeled + unlabeled data
                new_train_iter = DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True)
                if optimizer is None:
                    optimizer = BertAdam(optimizer_grouped_parameters,
                                         lr=golf_args.lr,
                                         warmup=golf_args.warmup_ratio,
                                         t_total=len(new_train_iter) * golf_args.epoch)
                # new_train_iter = build_iterator(train_dataset, ldsgm_config)
                utils.writelog("Beginning GOLF training with pseudo-labels.....")
                train_golf(golf_args, semisupervised_model, new_train_iter, dev_iter, test_iter, optimizer)

        ##################
        # /Training phase
        ##################

        ##################
        # Validation phase
        ##################
        # call LDSGM/GOLF validation with new model
        best_f1 = -1
        with torch.no_grad():
            for p_i in range(n_models):
                # do LDSGM validation only
                # get newly created model
                if da_config.model == "LDSGM":
                    loss_test, acc_top_test, f1_top_test, acc_sec_test, f1_sec_test, acc_conn_test, f1_conn_test \
                        = get_dev_scores("LDSGM", dev_iter, semisupervised_model, da_config,
                                         add_model_id_to_path(path_dev_pred, p_i))
                elif da_config.model == "GOLF":
                    loss_test, acc_top_test, f1_top_test, acc_sec_test, f1_sec_test, acc_conn_test, f1_conn_test \
                        = get_dev_scores("GOLF", dev_iter, semisupervised_model, golf_args,
                                         add_model_id_to_path(path_dev_pred, p_i))
                scores = metrics.attachment_scores(
                    pred_path=add_model_id_to_path(path_dev_pred, p_i),
                    gold_path=path_dev_gold, f1=f1_sec_test, accuracy=acc_sec_test)
                scores["f1"] *= 100.0
                scores["epoch"] = epoch
                # writer_dev.write(scores)
                utils.writelog(utils.pretty_format_dict(scores))

                did_update = bestscore_holders["independent"][p_i].compare_scores(scores["f1"], epoch)

                new_model = semisupervised_model
                # Save the model?
                if did_update:
                    torch.save(semisupervised_model.state_dict(), add_model_id_to_path(path_snapshot, p_i))
                    #    idrr_model_list[p_i].save_model(path=add_model_id_to_path(path_snapshot, p_i))
                    utils.writelog("Saved model_list[%d] to %s" % (p_i, add_model_id_to_path(path_snapshot, p_i)))
                if (da_config.bootstrapping_type != "at") or (da_config.bootstrapping_type == "at" and p_i == 0):
                    if best_f1 < scores["f1"]:
                        best_f1 = scores["f1"]

        bestscore_holders["joint"].compare_scores(best_f1, epoch)
        utils.writelog("[Epoch %d] Max validation F1 score: %f" % (epoch, bestscore_holders["joint"].best_score))

        # Finished?
        if bestscore_holders["joint"].ask_finishing(max_patience=10):
            utils.writelog("Patience %d is over. Training finished successfully." % bestscore_holders["joint"].patience)
            writer_train.close()
            writer_dev.close()
            return

        ##################
        # /Validation phase
        ##################

    ##################
    # /Training-and-validation loops
    ##################

    writer_train.close()
    writer_dev.close()


def select_pseudo_labeled_data_for_selftraining(
        unlabeled_dataset,
        ann_list_top,
        ann_list_sec,
        confidence_scores_list,
        topk_ratio):
    """Sample selection function for self-training

    Parameters
    ----------
    unlabeled_dataset: numpy.ndarray
    path_ann_list: list[str]
    topk_ratio: float

    Returns
    -------
    list[numpy.ndarray]
    dict[str, Any]
    """
    N_PARSERS = 1
    # assert len(path_ann_list) == N_PARSERS

    pool_size = len(unlabeled_dataset)

    pseudo_labeled_dataset_list = [[] for _ in range(N_PARSERS)]
    info = [{} for _ in range(N_PARSERS)]

    # Get indicators for the selected pseudo-labeled_ldsgm data
    indicators_list = []

    for teacher_i in range(N_PARSERS):
        indicators, info[teacher_i]["max_score"], info[teacher_i]["min_score"] \
            = rank_above_k(confidence_scores=confidence_scores_list,
                           topk=int(pool_size * topk_ratio))
        indicators_list.append(indicators)
    # print(indicators_list)

    # create dataset
    # f = open(path_ann, 'w', encoding='utf-8')
    # create the csv writer
    # writer = csv.writer(f)

    class_4_prediction_dict = {}
    class_11_prediction_dict = {}
    for c in class4:
        class_4_prediction_dict[c] = 0

    for c in class11:
        class_11_prediction_dict[c] = 0

    for data_i, data in enumerate(pyprind.prog_bar(unlabeled_dataset)):
        for student_i in range(N_PARSERS):
            teacher_i = student_i
            if indicators_list[teacher_i][data_i]:
                # teacher_arcs = batch_arcs_list[teacher_i][data_i]
                # create instance of pseudo labeled dataset
                pseudo_labeled_data = build_dataset(ldsgm_model.config, arg1=data[12]["arg1"], arg2=data[12]["arg2"],
                                                    label1=ann_list_top[data_i], label2=ann_list_sec[data_i])
                class_4_prediction_dict[class4[ann_list_top[data_i]]] += 1
                class_11_prediction_dict[class11[ann_list_sec[data_i]]] += 1
                # print(pseudo_labeled_data)

                pseudo_labeled_dataset_list[student_i].append(pseudo_labeled_data)
                # check index in annotation lists; map to class value
                # row = [data[12]["arg1"], data[12]["arg2"], class4[ann_list_top[data_i]], class11[ann_list_sec[data_i]]]
                # writer.writerow(row)

    writer_pred.write(class_4_prediction_dict)
    writer_pred.write(class_11_prediction_dict)
    # f.close()
    print(len(pseudo_labeled_dataset_list))
    return pseudo_labeled_dataset_list, info


def train_ldsgm(config, model, train_iter, dev_iter, test_iter, ldsgm_params):
    optimizer = ldsgm_params["optimizer"]
    optimizer_reverse = ldsgm_params["optimizer_reverse"]

    model.train()
    criterion_kl_loss = nn.KLDivLoss(reduction='batchmean')

    model.train()
    start_time = time.time()
    for i, (trains, y1, y2, argmask) in enumerate(train_iter):
        outputs_top, outputs_sec, outputs_conn, outputs_top_reverse, outputs_sec_reverse, outputs_conn_reverse = model(
            trains, argmask)

        model.zero_grad()

        loss_top = F.cross_entropy(outputs_top, y1[0])
        loss_sec = F.cross_entropy(outputs_sec, y1[1])
        # loss_conn = F.cross_entropy(outputs_conn, y1[2])

        loss_kl_top = criterion_kl_loss(torch.log_softmax(outputs_top, dim=-1),
                                        torch.softmax(outputs_top_reverse.detach(), dim=-1))
        loss_kl_sec = criterion_kl_loss(torch.log_softmax(outputs_sec, dim=-1),
                                        torch.softmax(outputs_sec_reverse.detach(), dim=-1))
        # loss_kl_conn = criterion_kl_loss(torch.log_softmax(outputs_conn, dim=-1),
        # torch.softmax(outputs_conn_reverse.detach(), dim=-1))
        # auxilary decoder loss
        loss_top_reverse = F.cross_entropy(outputs_top_reverse, y1[0])
        loss_sec_reverse = F.cross_entropy(outputs_sec_reverse, y1[1])
        # loss_conn_reverse = F.cross_entropy(outputs_conn_reverse, y1[2])
        loss_kl_top_reverse = criterion_kl_loss(torch.log_softmax(outputs_top_reverse, dim=-1),
                                                torch.softmax(outputs_top.detach(), dim=-1))
        loss_kl_sec_reverse = criterion_kl_loss(torch.log_softmax(outputs_sec_reverse, dim=-1),
                                                torch.softmax(outputs_sec.detach(), dim=-1))
        # loss_kl_conn_reverse = criterion_kl_loss(torch.log_softmax(outputs_conn_reverse, dim=-1),
        #                                         torch.softmax(outputs_conn.detach(), dim=-1))

        loss_kl = loss_kl_top + loss_kl_sec  # + loss_kl_conn
        loss_kl_reverse = loss_kl_top_reverse + loss_kl_sec_reverse  # + loss_kl_conn_reverse

        loss = loss_top * config.lambda1 + loss_sec * config.lambda2 + loss_kl * config.lambda4  # + loss_conn * config.lambda3
        loss_reverse = loss_top_reverse * config.lambda1 + loss_sec_reverse * config.lambda2 + loss_kl_reverse * config.lambda4  # + loss_conn_reverse * config.lambda3

        loss.backward(retain_graph=True)
        optimizer.step()
        loss_reverse.backward()
        optimizer_reverse.step()

        ldsgm_params["total_batch"] += 1
        if ldsgm_params["total_batch"] % 100 == 0:
            print(ldsgm_params["total_batch"])

        if config.show_time:
            if ldsgm_params["total_batch"] % 100 == 0:
                y_true_top = y1[0].data.cpu()
                y_true_sec = y1[1].data.cpu()
                y_true_conn = y1[2].data.cpu()
                if config.need_clc_loss:
                    y_predit_top = torch.max(outputs_top.data, 1)[1].cpu()
                    y_predit_sec = torch.max(outputs_sec.data, 1)[1].cpu()
                    y_predit_conn = torch.max(outputs_conn.data, 1)[1].cpu()

                else:
                    y_predit_top = outputs_top.data.cpu()
                    y_predit_sec = outputs_sec.data.cpu()

                train_acc_top = sklearn.metrics.accuracy_score(y_true_top, y_predit_top)
                train_acc_sec = sklearn.metrics.accuracy_score(y_true_sec, y_predit_sec)
                train_acc_conn = sklearn.metrics.accuracy_score(y_true_conn, y_predit_conn)

                loss_dev, acc_top, f1_top, acc_sec, f1_sec, acc_conn, f1_conn = evaluate(config, model, dev_iter)

                if (f1_top + f1_sec) > (
                        ldsgm_params["dev_best_f1_top"] + ldsgm_params["dev_best_f1_sec"]):
                    ldsgm_params["dev_best_f1_top"] = f1_top
                    ldsgm_params["dev_best_f1_sec"] = f1_sec
                    ldsgm_params["dev_best_f1_conn"] = f1_conn
                    ldsgm_params["dev_best_acc_top"] = acc_top
                    ldsgm_params["dev_best_acc_sec"] = acc_sec
                    ldsgm_params["dev_best_acc_conn"] = acc_conn
                    torch.save(model.state_dict(), config.save_path_top)
                    improve = '*'
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)

                msg = 'top-down:TOP: Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},' + \
                      'Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, Val F1: {5:>6.2%} Time: {6} {7}'
                lgg.info(msg.format(ldsgm_params["total_batch"], loss.item(), train_acc_top, loss_dev, acc_top, f1_top,
                                    time_dif,
                                    improve))
                msg = 'top-down:SEC: Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},' + \
                      'Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, Val F1: {5:>6.2%} Time: {6} {7}'
                lgg.info(msg.format(ldsgm_params["total_batch"], loss.item(), train_acc_sec, loss_dev, acc_sec, f1_sec,
                                    time_dif,
                                    improve))
                msg = 'top-down:CONN: Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},' + \
                      'Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, Val F1: {5:>6.2%} Time: {6} {7}'
                lgg.info(
                    msg.format(ldsgm_params["total_batch"], loss.item(), train_acc_conn, loss_dev, acc_conn, f1_conn,
                               time_dif,
                               improve))

                lgg.info(' ')

            model.train()

    time_dif = get_time_dif(start_time)
    lgg.info("Train time usage: {}".format(time_dif))
    # acc_top_test, f1_top_test, acc_sec_test, f1_sec_test, acc_conn_test, f1_conn_test \
    #   = test(config, model, test_iter)

    dev_msg = 'dev_best_acc_top: {0:>6.2%},  dev_best_f1_top: {1:>6.2%}, \n' + \
              'dev_best_acc_sec: {2:>6.2%},  dev_best_f1_sec: {3:>6.2%}, \n' + \
              'dev_best_acc_conn: {4:>6.2%},  dev_best_f1_conn: {5:>6.2%}'
    lgg.info(dev_msg.format(ldsgm_params["dev_best_acc_top"], ldsgm_params["dev_best_f1_top"],
                            ldsgm_params["dev_best_acc_sec"], ldsgm_params["dev_best_f1_sec"],
                            ldsgm_params["dev_best_acc_conn"], ldsgm_params["dev_best_f1_conn"]))


def train_golf(golf_args, g_model, train_loader, dev_loader, test_loader, optimizer):
    total_batch = 0
    dev_best_acc_top = 0.0
    dev_best_acc_sec = 0.0
    dev_best_acc_conn = 0.0
    dev_best_f1_top = 0.0
    dev_best_f1_sec = 0.0
    dev_best_f1_conn = 0.0

    start_time = time.time()
    for i, (x, _, mask, _, y1_top, y1_sec, y1_conn, _, _, _, arg1_mask, arg2_mask) in enumerate(train_loader):
        g_model.train()
        # print(y1_sec)
        logits_top, logits_sec, logits_conn, loss = g_model(x, mask, y1_top, y1_sec, y1_conn, arg1_mask, arg2_mask,
                                                          train=True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_batch += 1
        if total_batch % args.evaluate_steps == 0:
            print(total_batch)

            y_true_top = y1_top.data.cpu()  # (batch)
            y_true_sec = y1_sec.data.cpu()  # (batch)
            y_true_conn = y1_conn.data.cpu()  # (batch)

            y_predit_top = torch.max(logits_top.data, 1)[1].cpu()  # (batch)
            y_predit_sec = torch.max(logits_sec.data, 1)[1].cpu()  # (batch)
            y_predit_conn = torch.max(logits_conn.data, 1)[1].cpu()  # (batch)

            train_acc_top = sklearn.metrics.accuracy_score(y_true_top, y_predit_top)
            train_acc_sec = sklearn.metrics.accuracy_score(y_true_sec, y_predit_sec)
            train_acc_conn = sklearn.metrics.accuracy_score(y_true_conn, y_predit_conn)

            # evaluate
            loss_dev, acc_top, f1_top, acc_sec, f1_sec, acc_conn, f1_conn = training.evaluate(args, g_model, dev_loader)

            if (acc_top + acc_sec + acc_conn + f1_top + f1_sec + f1_conn) \
                    > (
                    dev_best_acc_top + dev_best_acc_sec + dev_best_acc_conn + dev_best_f1_top + dev_best_f1_sec + dev_best_f1_conn):
                dev_best_f1_top = f1_top
                dev_best_f1_sec = f1_sec
                dev_best_f1_conn = f1_conn
                dev_best_acc_top = acc_top
                dev_best_acc_sec = acc_sec
                dev_best_acc_conn = acc_conn
                # torch.save(model.state_dict(), args.save_file + args.model_name_or_path.split('/')[-1] + '.ckpt')
                improve = '*'
                last_improve = total_batch
            else:
                improve = ''
            time_dif = get_time_dif(start_time)

            msg = 'top-down:TOP: Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},' + \
                  'Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, Val F1: {5:>6.2%} Time: {6} {7}'
            lgg.info(
                msg.format(total_batch, loss.item(), train_acc_top, loss_dev, acc_top, f1_top, time_dif, improve))
            msg = 'top-down:SEC: Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},' + \
                  'Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, Val F1: {5:>6.2%} Time: {6} {7}'
            lgg.info(
                msg.format(total_batch, loss.item(), train_acc_sec, loss_dev, acc_sec, f1_sec, time_dif, improve))
            msg = 'top-down:CONN: Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},' + \
                  'Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, Val F1: {5:>6.2%} Time: {6} {7}'
            lgg.info(msg.format(total_batch, loss.item(), train_acc_conn, loss_dev, acc_conn, f1_conn, time_dif,
                                improve))

            lgg.info(' ')
            lgg.info(' ')

            # if total_batch - last_improve > args.require_improvement:
            # training stop
            #    lgg.info("No optimization for a long time, auto-stopping...")
            # flag = True
            #    break

    #time_dif = get_time_dif(start_time)
    #lgg.info("Train time usage: {}".format(time_dif))
    #acc_top_test, f1_top_test, acc_sec_test, f1_sec_test, acc_conn_test, f1_conn_test \
    #    = training.test(golf_args, g_model, test_loader)

    dev_msg = 'dev_best_acc_top: {0:>6.2%},  dev_best_f1_top: {1:>6.2%}, \n' + \
              'dev_best_acc_sec: {2:>6.2%},  dev_best_f1_sec: {3:>6.2%}, \n' + \
              'dev_best_acc_conn: {4:>6.2%},  dev_best_f1_conn: {5:>6.2%}'
    lgg.info(dev_msg.format(dev_best_acc_top, dev_best_f1_top,
                            dev_best_acc_sec, dev_best_f1_sec,
                            dev_best_acc_conn, dev_best_f1_conn))


def rank_above_k(confidence_scores, topk):
    """Sample selection criterion: Rank-above-k

    Parameters
    ----------
    confidence_scores: list[float]
    topk: int

    Returns
    -------
    numpy.ndarray(shape=(pool_size,), dtype=bool)
    int
    int
    """
    pool_size = len(confidence_scores)
    indicators = np.zeros((pool_size,))

    # Sort data according to confidence scores
    # print(confidence_scores)
    sorted_scores = [(c, i) for i, c in enumerate(confidence_scores)]
    # print(sorted_scores)
    sorted_scores = sorted(sorted_scores, key=lambda tpl: -tpl[0])
    # print(sorted_scores)

    # Select top-k highest-scoring data
    sorted_scores = sorted_scores[:topk]
    max_score = sorted_scores[0][0]
    min_score = sorted_scores[-1][0]

    # Get indices of the selected data
    selected_indices = [i for c, i in sorted_scores]
    assert len(selected_indices) == topk

    # Convert to indicators
    indicators[selected_indices] = 1
    indicators = indicators.astype(np.bool)

    return indicators, max_score, min_score


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# torch.use_deterministic_algorithms(True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # LDSGM args
    parser.add_argument('--model', type=str, default='LDSGM', help='choose a model')
    parser.add_argument('--tune', type=int, default=1, choices=[1, 0], help='fine tune or not: 0 or 1')
    parser.add_argument('--cuda', type=int, default=0, choices=[0, 1], help='choose a cuda: 0 or 1')
    parser.add_argument('--base', type=str, default='roberta', choices=['roberta'], help='roberta model as encoder')
    parser.add_argument('--lambda4', type=float, default=1.0, help='lambda for kl loss')

    print(os.getcwd())
    prefix = utils.get_current_time()
    base_dir = "LDSGM-main/DomainAdaptation"
    dataset = 'data/labeled_ldsgm'  # path where old model is saved (under 'saved-dict' directory)
    new_model_dataset = 'data/new_models'  # path where new semi-supervised model is saved (under 'saved-dict' directory)
    golf_dataset = 'GOLF_files/Ji'  # golf model data files

    unlabeled_dataset_path = 'data/unlabeled/annotations/sample'  # unlabeled data for bootstrapping

    # 2 GOLF model args
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    parser.add_argument('--data_file', type=str, default='GOLF_files/Ji/data/', help='the file of data')
    parser.add_argument('--log_file', type=str, default='GOLF_files/Ji/log/', help='the file of log')
    parser.add_argument('--save_file', type=str, default='GOLF_files/Ji/saved_dict/', help='save model file')

    ## model arguments
    parser.add_argument('--model_name_or_path', type=str, default='roberta-base', help='the name of pretrained model')
    parser.add_argument('--freeze_bert', action='store_true', default=False,
                        help='whether freeze the parameters of bert')
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature of contrastive learning')
    parser.add_argument('--num_co_attention_layer', type=int, default=2, help='number of co-attention layers')
    parser.add_argument('--num_gcn_layer', type=int, default=2, help='number of gcn layers')
    parser.add_argument('--gcn_dropout', type=float, default=0.1, help='dropout rate after gcn layer')
    parser.add_argument('--label_embedding_size', type=int, default=100, help='embedding dimension of labels')
    parser.add_argument('--lambda_global', type=float, default=0.1,
                        help='lambda for global_hierarcial_contrastive_loss')
    parser.add_argument('--lambda_local', type=float, default=1.0, help='lambda for local_hierarcial_contrastive_loss')
    ## training arguments
    parser.add_argument('--pad_size', type=int, default=100, help='the max sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epoch', type=int, default=15, help='training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='warmup_ratio')
    parser.add_argument('--evaluate_steps', type=int, default=100, help='number of evaluate_steps')
    parser.add_argument('--require_improvement', type=int, default=10000, help='early stop steps')
    args = parser.parse_args()

    # 1 LDSGM model configuration
    orig_config = Config(dataset, args.cuda, True, args.base, args.lambda4, unlabeled_dataset_path)
    ldsgm_config = Config(new_model_dataset, args.cuda, True, args.base, args.lambda4, unlabeled_dataset_path)
    # create labeled dataset and unlabeled dataset annotations using public annotations to be used for bootstrapping
    train_data, dev_data, test_data, unlabeled_data = build_dataset(ldsgm_config,
                                                                    semi_supervision=True)  # change to True
    ldsgm_dev_iter = build_iterator(dev_data, ldsgm_config)  # PDTB3 dev set
    ldsgm_test_iter = build_iterator(test_data, ldsgm_config)  # DiscoGem test set

    model_name = args.model
    # 1. load trained LDSGM base model
    x = import_module(model_name)
    ldsgm_model = x.Model(orig_config).to(orig_config.device)
    state_dict = torch.load("models/ldsgm_base/roberta_top.ckpt", map_location=torch.device("cpu"))
    remove_prefix = "module."
    state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
    # del state_dict['bert.embeddings.position_ids']
    ldsgm_model.load_state_dict(state_dict, strict=False)
    ldsgm_model.config = orig_config

    # 2. load trained GOLF base model
    args.i2top = [x.strip() for x in open(args.data_file + 'top.txt').readlines()]
    args.top2i = dict((x, xid) for xid, x in enumerate(args.i2top))
    args.n_top = len(args.i2top)
    args.i2sec = [x.strip() for x in open(args.data_file + 'sec.txt').readlines()]
    args.sec2i = dict((x, xid) for xid, x in enumerate(args.i2sec))
    args.n_sec = len(args.i2sec)
    args.i2conn = [x.strip() for x in open(args.data_file + 'conn.txt').readlines()]
    args.conn2i = dict((x, xid) for xid, x in enumerate(args.i2conn))
    args.n_conn = len(args.i2conn)
    args.label_num = args.n_top + args.n_sec + args.n_conn  # total label num(top:4,second:11,conn:186)

    args.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    args.config = AutoConfig.from_pretrained(args.model_name_or_path)

    args.t = datetime.now().strftime('%B%d-%H:%M:%S')
    args.log = args.log_file + str(args.t) + '.log'
    print(args.log)
    args.device = torch.device('cuda:{0}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

    # setlogging(lgg.DEBUG, 'args.log')
    seed_torch(args.seed)

    hyper_parameters = args.__dict__.copy()
    hyper_parameters['i2conn'] = ''
    hyper_parameters['conn2i'] = ''
    hyper_parameters['i2top'] = ''
    hyper_parameters['top2i'] = ''
    hyper_parameters['i2sec'] = ''
    hyper_parameters['sec2i'] = ''
    hyper_parameters['tokenizer'] = ''
    hyper_parameters['config'] = ''
    lgg.info(hyper_parameters)

    start_time = time.time()
    lgg.info("Loading data...")
    golf_config = GOLFConfig()
    x = import_module("GOLF")  # import GOLF module
    golf_model = x.Model(args).to(orig_config.device)
    print(os.getcwd())
    state_dict_2 = torch.load("models/golf_base/roberta-base.ckpt", map_location=torch.device("cpu"))
    golf_model.load_state_dict(state_dict_2, strict=False)  # load trained model params

    # Load GOLF datasets
    golf_train_dataset = MyDataset(args, args.data_file + 'train.txt')
    golf_dev_dataset = MyDataset(args, args.data_file + 'dev.txt')
    golf_test_dataset = MyDataset(args, args.data_file + 'test.txt')
    # golf_unlabeled_dataset = MyDataset(args, unlabeled_dataset_path + '/data/train.txt')
    # torch.set_printoptions(profile="full")

    golf_dev_loader = DataLoader(dataset=golf_dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False)
    golf_test_loader = DataLoader(dataset=golf_test_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False)

    # bootstrapping configuration
    da_config = DAConfig()

    # Training loss and etc.
    path_train_losses = os.path.join("results/", prefix + ".train.losses.jsonl")

    # Model snapshot
    path_snapshot = os.path.join("results/", prefix + ".model")

    # Automatic annotation on unlabeled data
    path_ann = os.path.join("results/", prefix + "_annotation.csv")

    # Validation outputs and scores
    path_dev_pred = os.path.join("results/", prefix + ".dev.pred.arcs")
    path_dev_eval = os.path.join("results/", prefix + ".dev.eval.jsonl")
    writer_pred = jsonlines.Writer(open(os.path.join("results/", prefix + ".predictions.eval.jsonl"), "a"), flush=True)

    # LDSGM/GOLF self-training (set model name in DAConfig class)
    if da_config.model == "LDSGM":
        train(
            da_config=da_config,
            idrr_model_list=[ldsgm_model],  # TODO remove comment later: this variable is parser_list var
            labeled_data=train_data,
            unlabeled_dataset=unlabeled_data,
            path_train_losses=path_train_losses,
            dev_iter=ldsgm_dev_iter,
            test_iter=ldsgm_test_iter,

            path_snapshot=path_snapshot,
            path_ann=path_ann,
            path_dev_pred=path_dev_pred,
            path_dev_gold=path_dev_pred,  # TODO add gold for unlabeled dataset that has gold data
            path_dev_eval=path_dev_eval)
    elif da_config.model == "GOLF":
        # GOLF self-training
        train(
            da_config=da_config,
            idrr_model_list=[golf_model],  # TODO remove comment later: this variable is parser_list var
            labeled_data=golf_train_dataset,
            unlabeled_dataset=unlabeled_data,
            path_train_losses=path_train_losses,
            dev_iter=golf_dev_loader,
            test_iter=golf_test_loader,
            path_snapshot=path_snapshot,
            path_ann=path_ann,
            path_dev_pred=path_dev_pred,
            path_dev_gold=path_dev_pred,  # TODO add gold for unlabeled dataset that has gold data
            path_dev_eval=path_dev_eval,
            golf_args=args)

    writer_pred.close()
