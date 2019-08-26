import re
import string

import nltk

from text_utils import read_file
from config import STOP_WORDS

"""
author: Silvio Severino
"""


class Sentences(object):
    """
    Class used to build an iterator to read input data from disk on-the-fly, necessary for memory.
    Moreover it does the necessary tasks in order to preprocess the sentences before the training.
    """

    def __init__(self, euro_dirname, sew_dirname, tom_dirname):
        """
        The instantiation of Sentence class
        :param euro_dirname: path of parsed Eurosense sentences
        :param sew_dirname: path of parsed SEW sentences
        :param tom_dirname: path of parsed TOM sentences
        """

        # datasets paths
        self.euro_dirname = euro_dirname
        self.sew_dirname = sew_dirname
        self.tom_dirname = tom_dirname

        # regex for HTML tag
        self.pattern = re.compile(r"&\w+;")

        # stopwords (~600)
        self.cachedStopWords = set(read_file(STOP_WORDS))

        # punctuations
        self.punctuation = set(string.punctuation)

        # it is possible to stemm, but it doesn't improve the score enought
        self.stemmer = nltk.stem.porter.PorterStemmer()

    def __iter__(self):
        """
        The iterator used to preprocess the sentences (from Eurosense, TOM and SEW) and to read
        from disk on-the-fly.
        In particular it transforms the sentences in lower case and it splits it. Then it removes
        the stopwords, the punctuations and some HTML tags.
        :return: a List of List of preprocessed sentences
        """
        for directory in [self.euro_dirname, self.sew_dirname, self.tom_dirname]:
            with open(directory, "r") as file:
                for line in file:

                    line = line.lower().split()

                    yield [
                        word
                        for word in line
                        if word not in self.cachedStopWords
                        and not self.pattern.match(word)
                        and word not in self.punctuation
                    ]
