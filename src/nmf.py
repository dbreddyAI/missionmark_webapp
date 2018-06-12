# Mark Evers
# 5/7/18
# nlp.py
# Script for model extraction


from sklearn.decomposition import NMF
import globals as g


def create_topic_model(corpus, n_topics):

    nmf = NMF(n_components=n_topics, max_iter=g.MAX_ITER, random_state=g.RANDOM_STATE)
    W = nmf.fit_transform(corpus.)