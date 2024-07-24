import csv
import sys
import pandas as pd
from pandas.errors import EmptyDataError

from DomainAdaptation.DomainAdapt import Genre


class EuroparlReader:
    def __init__(self, src_filename, src_genre: Genre):
        self.src_filename = src_filename
        try:
            self.df = pd.read_csv(src_filename, sep='\t', on_bad_lines='skip', quoting=csv.QUOTE_NONE)
        except EmptyDataError:
            print("error!")
            self.df = None
        self.genre = src_genre
        # print(self.df)

    def iter_data(self, display_progress=False):
        for i, row in self.df.iterrows():
            if display_progress:
                sys.stderr.write("\r")
                sys.stderr.write("row {}".format(i + 1))
                sys.stderr.flush()

            yield DataRow(row)
        if display_progress:
            sys.stderr.write("\n")


class DataRow:
    # NOTE: order of headers appearing in csv must be preserved for now
    header = ['A', 'B', 'C']

    def __init__(self, row):
        for i in range(len(row)):
            row_value = row[i]
            setattr(self, self.header[i], row_value)


######################################################################

if __name__ == '__main__':
    for datum in EuroparlReader('data/unlabeled/raw/europarl/cs_to_en.en.ep-08-12-03-019.csv', Genre.EUROPARL).iter_data():
        print(datum.B)
        print(datum.C)
