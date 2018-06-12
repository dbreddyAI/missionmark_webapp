# Mark Evers
# 5/7/18
# nlp.py
# Script for model extraction


from sklearn.decomposition import NMF
import globals as g
from pickle_workaround import pickle_load, pickle_dump


def create_topic_model(corpus, n_topics):

    g.debug(f"Extracting {n_topics} latent topics...")

    nmf = NMF(n_components=n_topics, max_iter=g.MAX_ITER, random_state=g.RANDOM_STATE)
    W = nmf.fit_transform(corpus.tfidf_corpus)

    g.debug(f" -> Finished in {nmf.n_iter_} iterations", 1)
    return nmf, W


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--topics", type=int, help="The number of latent topics to extract.")

    args = parser.parse_args()
    n_topics = args.topics if args.topics else g.N_TOPICS

    corpus = pickle_load("data/pickles/corpus.pkl")
    nmf, W = create_topic_model(corpus, n_topics)

    pickle_dump(nmf, "data/pickles/nmf.pkl")
    pickle_dump(W, "data/pickles/W.pkl")


if __name__ == "__main__":

    main()
