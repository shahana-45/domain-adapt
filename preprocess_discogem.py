from discogem_reader import DiscogemReader
import argparse
import os
import sys

top_senses = set(['Temporal', 'Comparison', 'Contingency', 'Expansion'])

selected_second_senses = {'Temporal.Asynchronous', 'Temporal.Synchronous', 'Contingency.Cause',
                          'Contingency.Cause+Belief', 'Contingency.Condition', 'Contingency.Purpose',
                          'Comparison.Contrast', 'Comparison.Concession', 'Expansion.Equivalence',
                          'Expansion.Conjunction', 'Expansion.Instantiation', 'Expansion.Level-of-detail',
                          'Expansion.Manner', 'Expansion.Substitution'}

sense_dict = {
    'Temporal.Synchronous': ['synchronous'],
    'Temporal.Asynchronous': ['precedence', 'succession'],
    'Contingency.Cause': ['reason', 'result'],
    'Contingency.Cause+Belief': ['result+belief', 'reason+belief'],
    'Contingency.Condition': ['arg1-as-cond', 'arg2-as-cond'],
    'Contingency.Purpose': ['arg1-as-goal', 'arg2-as-goal'],
    'Comparison.Contrast': ['contrast'],
    'Comparison.Concession': ['arg1-as-denier', 'arg2-as-denier'],
    'Expansion.Equivalence': ['equivalence'],
    'Expansion.Conjunction': ['conjunction'],
    'Expansion.Instantiation': ['arg1-as-instance', 'arg2-as-instance'],
    'Expansion.Level-of-detail': ['arg1-as-detail', 'arg2-as-detail'],
    'Expansion.Manner': ['arg1-as-manner', 'arg2-as-manner'],
    'Expansion.Substitution': ['arg2-as-subst']
}

# for instances without the second-level senses.txt
# In training phase: we directly set as the most likely sub-sense of their top-level sense, respectively
# In testing phase: these instances will be discarded for calculate the acc and F_1 score at the second-level
top_2_second = {'Temporal': 'Temporal.Asynchronous',
                'Comparison': 'Comparison.Contrast',
                'Contingency': 'Contingency.Cause',
                'Expansion': 'Expansion.Conjunction'}


def preprocess(splitting):
    # following Ji, for 4-way and 11-way
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

    reader = DiscogemReader('./raw/DiscogemCorpus_annotations.csv')
    for corpus in reader.iter_data():
        #if corpus.genre != 'wikipedia':  # genre-wise data generation
        #    continue
        sense_l3 = corpus.majoritylabel_sampled
        extended_sense = None
        for key, values in sense_dict.items():
            if sense_l3 in values:
                extended_sense = key.strip() + '.' + sense_l3
                break

        if extended_sense:
            sense_split = extended_sense.split('.')
            sense_l2 = '.'.join(sense_split[0:2])
            arg1 = corpus.arg1
            arg2 = corpus.arg2

            if corpus.split == "train":
                #arg1_train.append(corpus.genre + " " + arg1)
                arg2_train.append(arg2)
                sense1_train.append([sense_split[0], sense_l2, corpus.domconn_step1])
                sense2_train.append([None, None, None])
            elif corpus.split == "dev":
                arg1_dev.append(arg1)
                arg2_dev.append(arg2)
                sense1_dev.append([sense_split[0], sense_l2, corpus.domconn_step1])
            elif corpus.split == "test":
                arg1_test.append(arg1)
                arg2_test.append(arg2)
                sense1_test.append([sense_split[0], sense_l2, corpus.domconn_step1])
            else:
                continue

            if corpus.domconn_step2:
                sense_l3 = corpus.majoritylabel_sampled
                extended_sense = None
                for key, values in sense_dict.items():
                    if sense_l3 in values:
                        extended_sense = key.strip() + '.' + sense_l3
                        break

                if extended_sense:
                    sense_l2 = '.'.join(sense_split[0:2])
                    if corpus.split == "train":
                        #arg1_train.append(corpus.genre + " " + arg1)
                        arg2_train.append(arg2)
                        sense1_train.append([sense_split[0], sense_l2, corpus.domconn_step2])
                        sense2_train.append([None, None, None])
                    elif corpus.split == "dev":
                        sense2_dev.append([sense_split[0], sense_l2, corpus.domconn_step2])
                    elif corpus.split == "test":
                        sense2_test.append([sense_split[0], sense_l2, corpus.domconn_step2])
                else:
                    sense_l2 = sense_l2 if sense_l2 in selected_second_senses else top_2_second[sense_split[0]]
                    if corpus.split == "train":
                        #arg1_train_other.append(corpus.genre + " " + arg1)
                        arg2_train_other.append(arg2)
                        sense1_train_other.append([sense_split[0], sense_l2, corpus.domconn_step2])
                        sense2_train_other.append([None, None, None])
                    elif corpus.split == "dev":
                        arg1_dev_other.append(arg1)
                        arg2_dev_other.append(arg2)
                        sense1_dev_other.append([sense_split[0], sense_l2, corpus.domconn_step2])
                    elif corpus.split == "test":
                        arg1_test_other.append(arg1)
                        arg2_test_other.append(arg2)
                        sense1_test_other.append([sense_split[0], sense_l2, corpus.domconn_step2])
                    else:
                        continue
            else:
                if corpus.split == "dev":
                    sense2_dev.append([None, None, None])
                elif corpus.split == "test":
                    sense2_test.append([None, None, None])
        else:
            # instances need special processing
            tmp_i += 1

            arg1 = corpus.arg1
            arg2 = corpus.arg2
            sense_l2 = top_2_second[sense_split[0]]

            if corpus.split == "train":
                #arg1_train_other.append(corpus.genre + " " + arg1)
                arg2_train_other.append(arg2)
                sense1_train_other.append([sense_split[0], sense_l2, corpus.domconn_step1])
                sense2_train_other.append([None, None, None])
            # elif corpus.Section in dev_sec: # TODO uncomment
            elif corpus.split == "dev":
                arg1_dev_other.append(arg1)
                arg2_dev_other.append(arg2)
                sense1_dev_other.append([sense_split[0], sense_l2, corpus.domconn_step1])
            # elif corpus.Section in test_sec: # TODO uncomment
            elif corpus.split == "test":
                arg1_test_other.append(arg1)
                arg2_test_other.append(arg2)
                sense1_test_other.append([sense_split[0], sense_l2, corpus.domconn_step1])
            else:
                continue

            if corpus.domconn_step2:
                sense_l3 = corpus.majoritylabel_sampled
                extended_sense = None
                for key, values in sense_dict.items():
                    if sense_l3 in values:
                        extended_sense = key.strip() + '.' + sense_l3
                        break

                if extended_sense:
                    sense_l2 = '.'.join(sense_split[0:2])
                    sense_l2 = sense_l2 if sense_l2 in selected_second_senses else top_2_second[sense_split[0]]

                if corpus.split == "train":
                    #arg1_train_other.append(corpus.genre + " " + arg1)
                    arg2_train_other.append(arg2)
                    sense1_train_other.append([sense_split[0], sense_l2, corpus.domconn_step2])
                    sense2_train_other.append([None, None, None])
                elif corpus.split == "dev":
                    sense2_dev_other.append([sense_split[0], sense_l2, corpus.domconn_step2])
                elif corpus.split == "test":
                    sense2_test_other.append([sense_split[0], sense_l2, corpus.domconn_step2])
            else:
                if corpus.split == "dev":
                    sense2_dev_other.append([None, None, None])
                elif corpus.split == "test":
                    sense2_test_other.append([None, None, None])

    print(tmp_i)
    # combined two parts of data
    arg1_train.extend(arg1_train_other)
    arg2_train.extend(arg2_train_other)
    sense1_train.extend(sense1_train_other)
    sense2_train.extend(sense2_train_other)

    arg1_dev.extend(arg1_dev_other)
    arg2_dev.extend(arg2_dev_other)
    sense1_dev.extend(sense1_dev_other)
    sense2_dev.extend(sense2_dev_other)

    arg1_test.extend(arg1_test_other)
    arg2_test.extend(arg2_test_other)
    sense1_test.extend(sense1_test_other)
    sense2_test.extend(sense2_test_other)

    assert len(arg1_train) == len(arg2_train) == len(sense1_train) == len(sense2_train)
    assert len(arg1_dev) == len(arg2_dev) == len(sense1_dev) == len(sense2_dev)
    assert len(arg1_test) == len(arg2_test) == len(sense1_test) == len(sense2_test)
    print('train size:', len(arg1_train))
    print('dev size:', len(arg1_dev))
    print('test size:', len(arg1_test))

    if splitting == 1:
        pre = './PDTB//genre_prepend//data//'

    with open(pre + 'train.txt', 'w', encoding='utf-8') as f:
        for arg1, arg2, sense1, sense2 in zip(arg1_train, arg2_train, sense1_train, sense2_train):
            print('{} ||| {} ||| {} ||| {}'.format(sense1, sense2, arg1, arg2), file=f)
    with open(pre + 'dev.txt', 'w', encoding='utf-8') as f:
        for arg1, arg2, sense1, sense2 in zip(arg1_dev, arg2_dev, sense1_dev, sense2_dev):
            print('{} ||| {} ||| {} ||| {}'.format(sense1, sense2, arg1, arg2), file=f)
    with open(pre + 'test.txt', 'w', encoding='utf-8') as f:
        for arg1, arg2, sense1, sense2 in zip(arg1_test, arg2_test, sense1_test, sense2_test):
            print('{} ||| {} ||| {} ||| {}'.format(sense1, sense2, arg1, arg2), file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='func', choices=['pre', 'test'], type=str, default='pre')
    parser.add_argument('-s', dest='splitting', choices=[1], type=int,
                        default='1')  # 1 for DiscoGem test set
    A = parser.parse_args()
    preprocess(A.splitting)
