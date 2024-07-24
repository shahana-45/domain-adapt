import csv
import sys
import pandas as pd


class DiscogemReader:
    def __init__(self, src_filename):
        self.src_filename = src_filename
        self.df = pd.read_csv(src_filename, usecols=DataRow.header)
        # print(self.df)

    def iter_data(self, display_progress=False):
        for i, row in self.df.iterrows():
            if display_progress:
                sys.stderr.write("\r")
                sys.stderr.write("row {}".format(i + 1))
                sys.stderr.flush()
            # print(row)
            # print('\n')
            yield DataRow(row)
        if display_progress:
            sys.stderr.write("\n")


class DataRow:
    # NOTE: order of headers appearing in csv must be preserved for now
    header = [
        'split', 'itemid', 'genre', 'majoritylabel_sampled', 'domconn_step1', 'domconn_step2', 'arg1', 'arg2'
    ]

    def __init__(self, row):
        for i in range(len(row)):
            att_name = DataRow.header[i]

            row_value = row[i]
            setattr(self, att_name, row_value)


######################################################################

if __name__ == '__main__':
    for datum in DiscogemReader('./raw/DiscogemCorpus_annotations.csv').iter_data():
        print(datum.arg1)