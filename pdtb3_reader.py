import csv
import sys

from nltk import WordNetLemmatizer


class CorpusReader:
    def __init__(self, src_filename):
        self.src_filename = src_filename

    def iter_data(self, display_progress=True):
        with open(self.src_filename) as f:
            row_iterator = csv.reader(f, delimiter='|')
            next(row_iterator)  # Skip past the header.
            for i, row in enumerate(row_iterator):
                if display_progress:
                    sys.stderr.write("\r")
                    sys.stderr.write("row {}".format(i + 1))
                    sys.stderr.flush()
                #print(row)
                #print('\n')
                yield Datum(row)
            if display_progress: sys.stderr.write("\n")


######################################################################

class Datum:
    header = [
        # Corpus
        'RelationType', 'ConnSpanList', 'ConnSrc',
        'ConnType', 'ConnPol', 'ConnDet', 'ConnFeatSpanList', 'Conn1',
        'SClass1A', 'SClass1B', 'Conn2', 'SClass2A', 'SClass2B',
        'Sup1SpanList', 'Arg1SpanList', 'Arg1SrcFirst', 'Arg1Type',
        'Arg1Pol', 'Arg1Det', 'Arg1FeatSpanList', 'Arg2SpanList',
        'Arg2Src', 'Arg2Type', 'Arg2Pol', 'Arg2Det', 'Arg2FeatSpanList',
        'Sup2SpanList', 'AdjuReason', 'AdjuDisagr', 'PBRole', 'PBVerb',
        'Offset', 'Provenance', 'Link', 'Conn', 'Arg1', 'Arg2', 'wsj_section',
        'wsj_doc'
    ]

    def __init__(self, row):
        for i in range(len(row)):
            att_name = Datum.header[i]
            row_value = row[i]
            setattr(self, att_name, row_value)

    def arg1_words(self, lemmatize=False): # TODO do we need to lemmatize?
        """
        Returns the list of words associated with Arg1. lemmatize=True
        uses nltk.stem.WordNetStemmer() on the list.
        """
        return self.__words(self.arg1_pos, lemmatize=lemmatize)

    def __words(self, method, lemmatize=False):
        """
        Internal method used by the X_words functions to get at their
        (possibly stemmed) words.
        """
        lemmas = method(lemmatize=lemmatize)
        return [x[0] for x in lemmas]

    def __lemmatize(self, lemma):
        """
        Internal method used for applying the nltk.stem.WordNetStemmer() to the (word, pos) pair lemma.
        """
        string, tag = lemma
        if tag in ('a', 'n', 'r', 'v'):
            wnl = WordNetLemmatizer()
            string = wnl.lemmatize(string, tag)
        return string, tag
