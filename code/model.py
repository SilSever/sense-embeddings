import logging

from gensim.models import Word2Vec, KeyedVectors

from config import EURO_SENTENCES, SEW_SENTENCES, EMBEDDINGS, TOM_SENTENCES, MODEL
from preprocesser import Sentences

"""
author: Silvio Severino
"""


def build_model():
    """
    This method is used to build the Word2Vec model using Gensim implementation.
    :return: word2vec embedding
    """

    # the iterator for the input sentences
    sentence = Sentences(EURO_SENTENCES, SEW_SENTENCES, TOM_SENTENCES)

    # log the training
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

    model = Word2Vec(
        sentence, min_count=3, workers=4, iter=5, size=400, window=5, hs=1, sample=1e-3)

    # model = FastText(
    #   sentence, min_count=3, workers=4, iter=5, size=400, window=5, hs=1, sample=1e-3)

    model.wv.save_word2vec_format(EMBEDDINGS, binary=False)
    return KeyedVectors.load_word2vec_format(EMBEDDINGS, binary=False)
