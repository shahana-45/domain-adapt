import pandas as pd

from DomainAdaptation.annotation_reader import AnnotationReader
import argparse
import os
import sys


def preprocess(splitting, genre, file_dir):
    arg1 = []
    arg2 = []
    tmp_i = 0
    os.chdir(sys.path[0])
    exclude_files = []

    # skip data present in discogem
    if genre == "wiki":
        exclude_files = pd.read_csv("../genres/discogem_wiki.csv", usecols=['datasource'], encoding="utf-8")

    excluded_files_list = exclude_files['datasource'].tolist()
    excluded_files_list = [f.split('_', 1)[1] for f in excluded_files_list]
    print(excluded_files_list)
    # read raw data files from directory
    directory = os.fsencode(file_dir)
    for file_name in os.listdir(directory):
        # print(file_name)
        if str(file_name).split('.')[1] in excluded_files_list:
            print(f"Excluded file {file_name} as it is present in DiscoGem!")
            continue
        reader = AnnotationReader(os.path.join(file_dir, file_name.decode("utf-8")), genre)
        if reader.df is None:  # some problem in reading the file
            print(f"No columns to parse from file {file_name}")
            continue

        for corpus in reader.iter_data():
            tmp_i += 1
            if tmp_i > 212509:
                break
            arg1.append(corpus.E)
            arg2.append(corpus.F)
            # instances need special processing

            # print(tmp_i)
            # combined two parts of data

    assert len(arg1) == len(arg2)
    print('test size:', len(arg1))

    if splitting == 1:
        pre = 'data/unlabeled/annotations/wiki_euro/'



    with open(pre + 'data/train.txt', 'a', encoding='utf-8') as f:
        for arg1, arg2 in zip(arg1, arg2):
            print('{} ||| {} ||| {} ||| {}'.format([None, None, None], [None, None, None], arg1, arg2), file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', dest='genre', choices=['wiki', 'europarl', 'novel'], type=str, default='wiki')
    parser.add_argument('-s', dest='splitting', choices=[1], type=int,
                        default='1')  # 1 for annotations
    A = parser.parse_args()
    preprocess(A.splitting, A.genre, 'data/unlabeled/raw/wiki/')
