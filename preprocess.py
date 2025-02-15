from pdtb3_reader import CorpusReader
import argparse
import os
import sys

top_senses = set(['Temporal', 'Comparison', 'Contingency', 'Expansion'])
'''
selected_second_senses = set([
    'Temporal.Asynchronous', 'Temporal.Synchrony', 'Contingency.Cause',
    'Contingency.Cause_belief', 'Comparison.Contrast', 'Comparison.Concession',
    'Expansion.Conjunction', 'Expansion.Instantiation', 'Expansion.Equivalence',
    'Expansion.Alternative', 'Expansion.List'
])
'''
selected_second_senses = {'Temporal.Asynchronous', 'Temporal.Synchronous', 'Contingency.Cause',
                          'Contingency.Cause+Belief', 'Contingency.Condition', 'Contingency.Purpose',
                          'Comparison.Contrast', 'Comparison.Concession', 'Expansion.Equivalence',
                          'Expansion.Conjunction', 'Expansion.Instantiation', 'Expansion.Level-of-detail',
                          'Expansion.Manner', 'Expansion.Substitution'}

# for instances without the second-level senses.txt
# In training phase: we directly set as the most likely sub-sense of their top-level sense, respectively
# In testing phase: these instances will be discarded for calclurate the acc and F_1 score at the second-level
top_2_second = {'Temporal': 'Temporal.Asynchronous',
                'Comparison': 'Comparison.Contrast',
                'Contingency': 'Contingency.Cause',
                'Expansion': 'Expansion.Conjunction'}


def arg_filter(input):
    arg = []
    pos = []
    for w in input:
        if w[1].find('-') == -1:
            arg.append(w[0].replace('\/', '/'))
            pos.append(w[1])
    return arg, pos


def preprocess(splitting):
    # following Ji, for 4-way and 11-way
    if splitting == 1:
        train_sec = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
                     '12', '13', '14', '15', '16', '17', '18', '19', '20']
        dev_sec = ['0', '1']
        test_sec = ['21', '22']

    # following Lin, for 4-way and 11-way
    elif splitting == 2:
        train_sec = [
            '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
            '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
        ]
        dev_sec = ['22']
        test_sec = ['23']

    # instances in 'selected_second_senses'
    arg1_train = []
    arg2_train = []
    sense1_train = []  # top, second, connective
    sense2_train = []  # None, None, None

    arg1_dev = []
    arg2_dev = []
    sense1_dev = []
    sense2_dev = []

    arg1_test = []
    arg2_test = []
    sense1_test = []
    sense2_test = []

    # other instances
    arg1_train_other = []
    arg2_train_other = []
    sense1_train_other = []  # top, second, connective
    sense2_train_other = []  # None, None, None

    arg1_dev_other = []
    arg2_dev_other = []
    sense1_dev_other = []
    sense2_dev_other = []

    arg1_test_other = []
    arg2_test_other = []
    sense1_test_other = []
    sense2_test_other = []

    tmp_i = 0
    os.chdir(sys.path[0])
    for corpus in CorpusReader('./raw/PDTB3.csv').iter_data():
        if corpus.RelationType != 'Implicit':  # train for implicit relations
            continue
            sense_split = corpus.ConnHeadSemClass1.split('.')
            sense_l2 = '.'.join(sense_split[0:2])
            if sense_l2 in selected_second_senses:
                arg1 = corpus.Arg1
                arg2 = corpus.Arg2
                if str(corpus.Section) in train_sec:
                    arg1_train.append(arg1)
                    arg2_train.append(arg2)
                    sense1_train.append([sense_split[0], sense_l2, corpus.Conn1])
                    sense2_train.append([None, None, None])
                elif str(corpus.Section) in dev_sec:
                    arg1_dev.append(arg1)
                    arg2_dev.append(arg2)

                    sense1_dev.append([sense_split[0], sense_l2, corpus.Conn1])
                elif str(corpus.Section) in test_sec:
                    arg1_test.append(arg1)
                    arg2_test.append(arg2)
                    sense1_test.append([sense_split[0], sense_l2, corpus.Conn1])

                else:
                    continue
                # Make sure to have same CSV fields. e.g ConnHeadSemClass1, Section, Conn2SemClass1
                if str(corpus.Conn2) != '':
                    sense_split = corpus.Conn2SemClass1.split('.')
                    sense_l2 = '.'.join(sense_split[0:2])
                    if sense_l2 in selected_second_senses:
                        if str(corpus.Section) in train_sec:
                            arg1_train.append(arg1)
                            arg2_train.append(arg2)
                            sense1_train.append(
                                [sense_split[0], sense_l2, corpus.Conn2])
                            sense2_train.append([None, None, None])
                        elif str(corpus.Section) in dev_sec:
                            sense2_dev.append(
                                [sense_split[0], sense_l2, corpus.Conn2])
                        elif str(corpus.Section) in test_sec:
                            sense2_test.append(
                                [sense_split[0], sense_l2, corpus.Conn2])
                else:
                    if str(corpus.Section) in dev_sec:
                        sense2_dev.append([None, None, None])
                    elif str(corpus.Section) in test_sec:
                        sense2_test.append([None, None, None])

        assert len(arg1_train) == len(arg2_train) == len(
            sense1_train) == len(sense2_train)
        assert len(arg1_dev) == len(arg2_dev) == len(sense1_dev) == len(sense2_dev)
        assert len(arg1_test) == len(arg2_test) == len(
            sense1_test) == len(sense2_test)
        print('train size:', len(arg1_train))
        print('dev size:', len(arg1_dev))
        print('test size:', len(arg1_test))

        if splitting == 1:
            pre = './PDTB3/Ji//data//'
        elif splitting == 2:
            pre = './PDTB3/Lin//data//'

        with open(pre + 'train.txt', 'w') as f:
            for arg1, arg2, sense1, sense2 in zip(arg1_train, arg2_train, sense1_train, sense2_train):
                print('{} ||| {} ||| {} ||| {}'.format(
                    sense1, sense2, ' '.join(arg1.split()), ' '.join(arg2.split())), file=f)
        with open(pre + 'dev.txt', 'w') as f:
            for arg1, arg2, sense1, sense2 in zip(arg1_dev, arg2_dev, sense1_dev, sense2_dev):
                print('{} ||| {} ||| {} ||| {}'.format(
                    sense1, sense2, ' '.join(arg1.split()), ' '.join(arg2.split())), file=f)
        with open(pre + 'test.txt', 'w') as f:
            for arg1, arg2, sense1, sense2 in zip(arg1_test, arg2_test, sense1_test, sense2_test):
                print('{} ||| {} ||| {} ||| {}'.format(
                    sense1, sense2, ' '.join(arg1.split()), ' '.join(arg2.split())), file=f)

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('-f', dest='func',
                            choices=['pre', 'test'], type=str, default='pre')
        # 1 for 'Ji', 2 for 'Lin'
        parser.add_argument('-s', dest='splitting',
                            choices=[1, 2], type=int, default='1')
        A = parser.parse_args()
        preprocess(A.splitting)