import csv

from gensim.models import KeyedVectors
from scipy.stats import spearmanr

from config import EMBEDDINGS, WORDSIM
from text_utils import load_lemma2syns

"""
author: Silvio Severino
"""


def is_oov(w1, w2, sensemb1, sensemb2, model, score):
    """
    This method is used to check if w1 or w2 are out-of-vocabulary (oov) words.
    It particular it checks whether they have not sense embeddings (retrieved from the dictionary)
    or if they are not in the model vocabulary.
    :param w1: first one lemma
    :param w2: second one lemma
    :param sensemb1: sense embedding belongs to w1
    :param sensemb2: sense embedding belongs to w2
    :param model: word2vec embeddings
    :param score: score
    :return: it returns None if w1 and w2 are not oov,
             score and one of the two lemmas (w1 or w2) otherwise.
    """

    if len(sensemb1) == 0 or all(senses not in model.vocab for senses in sensemb1):
        return score, w1.lower()
    elif len(sensemb2) == 0 or all(senses not in model.vocab for senses in sensemb2):
        return score, w2.lower()
    else:
        return None


def compute_sub_cos_sim(w1, w2, lemma2syns, model):
    """
    This method is used to compute the cosine similarity between two
    lemmas given in input.
    :param w1: first one lemma
    :param w2: second one lemma
    :param lemma2syns: map lemma to Babelnet synsets
    :param model: model embeddings
    :return: it returns score and an empty set if w1 and w2 are not oov,
             score and one of the two lemmas (w1 or w2) otherwise.
    """
    score = -1.0

    sensemb1 = lemma2syns[w1.lower()]
    sensemb2 = lemma2syns[w2.lower()]

    oov = is_oov(w1, w2, sensemb1, sensemb2, model, score)
    if oov is not None:
        return oov

    for s1 in sensemb1:
        for s2 in sensemb2:
            if model.vocab.get(s1) and model.vocab.get(s2):
                score = max(score, model.similarity(s1, s2))

    return score, ()


def compute_cos_sim(model, lemma2syn, path=WORDSIM):
    """
    This method is used to handle the cosine similarity computing
    :param model: model embeddings vectors
    :param lemma2syn: map lemma to Babelnet synsets
    :param path: test set path, by default WordSimilarity-353
    :return: spearman correlation between WordSimilarity-353 combined.tab gold and
             the cosine similarity computed
    """

    print("Computing cosine similarity...")
    human, cosine = [], []
    oov = set()

    with open(path, newline="") as file:
        file_reader = csv.reader(file, delimiter="\t")
        next(file_reader)

        for row in file_reader:
            res, tmp = compute_sub_cos_sim(row[0], row[1], lemma2syn, model)
            oov.add(tmp)

            human.append(float(row[2]))
            cosine.append(res)

    print("Out of vocabulary: ", len(oov) - 1)
    return spearmanr(human, cosine)


def main():

    lemma2syn = load_lemma2syns()
    model = KeyedVectors.load_word2vec_format(EMBEDDINGS, binary=False)

    cov = compute_cos_sim(model, lemma2syn)

    print(cov)


if __name__ == "__main__":
    main()
