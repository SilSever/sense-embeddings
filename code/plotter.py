import re

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

"""
author: Silvio Severino
"""


def preprocessing_visualizzation(model, lemma):

    pattern = re.compile("^" + lemma + "_bn:")
    keys = [sense for sense in model.vocab if pattern.match(sense)]

    embedding_clusters = []
    word_clusters = []
    for sense in keys:
        embeddings = [model[sense]]
        words = [sense]
        print(sense, "-->")
        for similar_word, _ in model.most_similar(sense, topn=18):
            words.append(similar_word)
            embeddings.append(model[similar_word])
        embedding_clusters.append(embeddings)
        word_clusters.append(words)

    return keys, np.array(embedding_clusters), word_clusters


def build_tsne_model(embedding_clusters):

    n, m, k = embedding_clusters.shape

    tsne_model_en_2d = TSNE(
        perplexity=50, n_components=2, init="pca", n_iter=6000, random_state=40
    )
    embeddings_en_2d = np.array(
        tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))
    ).reshape(n, m, 2)
    return embeddings_en_2d


def tsne_plot_similar_words(
    title, labels, embedding_clusters, word_clusters, a, filename=None
):

    plt.figure(figsize=(9, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    # colors = {'red', 'blue'}
    for label, embeddings, words, color in zip(
        labels, embedding_clusters, word_clusters, colors
    ):

        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(
                word,
                alpha=0.5,
                xy=(x[i], y[i]),
                xytext=(5, 2),
                textcoords="offset points",
                ha="right",
                va="bottom",
                size=8,
            )
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format="png", dpi=150, bbox_inches="tight")
    plt.show()
