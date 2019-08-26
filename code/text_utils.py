import os
from collections import defaultdict

from tqdm import tqdm

from config import EURO_SENTENCES, SEW_SENTENCES, TOM_SENTENCES, BN2WN_MAP, LEMMA2SYNS_MAP

"""
:author: Silvio Severino
"""


def build_bn2wns(path=BN2WN_MAP):
    """
    This method is used to build the map Babelnet synset to Wordet synsets
    :param path: path of the mapping file
    :return: a map Babelnet synset to List of Wordnet synsets
    """
    bn2wns = {}
    with open(path, mode="r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            ids = line.split("\t")

            # the first elem of the splitted line is the Babelnet synset,
            # the remaining is the Wordnet synsets list
            bn2wns[ids[0]] = ids[1:]

    return bn2wns


def build_wns2bn(path=BN2WN_MAP):
    """
    This method is used to build the map Wordnet synset to Babelnet synset
    :param path: path of the mapping file
    :return: a map Wordnet synset to Babelnet synset
    """
    wns2bn = {}
    with open(path, mode='r') as file:
        for line in file:
            line = line.strip()
            ids = line.split('\t')

            for wn in ids[1:]:
                wns2bn[wn] = ids[0]

    return wns2bn


def write_file(to_write, filename):
    """
    This method is used to write a list in a file
    :param to_write: List to write
    :param filename: path where write
    :return: None
    """
    file = open(filename, "a", encoding="utf8")
    for i in to_write:
        if i is not None:
            file.write(i + "\n")

    file.close()


def read_file(path):
    """
    This method is used to read a list from a file
    :param path: path from read
    :return: a list of sentences
    """
    file = []
    with open(path, mode="r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            file.append(line)

    return file


def save_lemma2syns(lemma2syns, path=LEMMA2SYNS_MAP):
    """
    This method is used to save the map lemma to Babelnet synsets in a file
    :param lemma2syns: map lemma to Babelnet synsets
    :param path: path where write
    :return: None
    """
    with open(path, mode="w") as file:
        for lemma in lemma2syns:
            file.write(lemma + " " + " ".join([syn for syn in lemma2syns[lemma]]) + "\n")


def load_lemma2syns(path=LEMMA2SYNS_MAP):
    """
    This method is used to load the map lemma to Babelnet synsets map from a file
    :param path: path from read
    :return: a map lemma to Babelnet synsets
    """

    # if the map doesn't exists
    if not os.path.isfile(path):
        return build_lemma2syns(build_bn2wns(BN2WN_MAP))

    lemma2syns = defaultdict(set)
    with open(LEMMA2SYNS_MAP, mode="r") as file:

        for line in file:
            line = line.strip().split(" ")

            lemma2syns[line[0]] = lemma2syns[line[0]] | set(line[1:])

    return lemma2syns


def build_lemma2syns(bn2wns, paths=[EURO_SENTENCES, SEW_SENTENCES, TOM_SENTENCES]):
    """
    This method is used to build the map lemma to Babelnet synsets.
    Note:
        it maps only the Babelnet synsets in the map
    :param bn2wns: map Babelnet synset to Wordnet synsets
    :param paths: path where save
    :return: a map lemma to Babelnet synsets
    """
    lemma2syns = defaultdict(set)

    for path in paths:

        with open(path, mode="r") as file:
            for line in tqdm(file, desc=('Parsing ' + path)):

                for lemma2syn in (word.lower() for word in line.strip().split() if "_bn:" in word):

                    splitted = lemma2syn.split("_")
                    if bn2wns.get(splitted[-1]):
                        lemma2syns["_".join(splitted[:-1])].add(lemma2syn)

    save_lemma2syns(lemma2syns)
    return lemma2syns
