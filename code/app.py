from parser import parse_all
from model import build_model
from text_utils import load_lemma2syns
from evaluator import compute_cos_sim

"""
author: Silvio Severino
"""


def main():
    
    parse_all()

    model = build_model()
    # model = KeyedVectors.load_word2vec_format(EMBEDDINGS, binary=False)

    lemma2syns = load_lemma2syns()
    cov = compute_cos_sim(model, lemma2syns)

    print(cov)


if __name__ == "__main__":
    main()
