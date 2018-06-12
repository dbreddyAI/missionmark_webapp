import globals as g
from wordcloud import WordCloud
import numpy as np



def build_word_cloud(vocab_weights, vocabulary):

    g.debug(f"Building wordcloud...", 2)

    wc = WordCloud(background_color=g.BACKGROUND_COLOR, max_words=g.MAX_WORDS, width=g.WIDTH, height=g.HEIGHT)
    wc.fit_words \
        ({vocabulary[word_i]: vocab_weights[word_i] for word_i in range(len(vocab_weights)) if vocab_weights[word_i]})

    return wc


def cache_wordclouds(corpus, vocabulary, H, W):

    n_topics = H.shape[0]
    topic_tfidf_weights = get_tfidf_topic_weights(corpus.tfidf_corpus)

    for topic_i in range(n_topics):
        # nmf wordcloud
        wc = build_word_cloud(H[topic_i], vocabulary)
        wc.to_file(f"output/wordclouds/{str(topic_i).rjust(3, '0')}_nmf.png")

        # tfidf wordcloud
        if topic_tfidf_weights[topic_i].sum():
            wc = build_word_cloud(topic_tfidf_weights[topic_i], vocabulary)
        else:
            # an empty topic...
            wc = build_word_cloud([1], ["This topic was empty"])
        wc.to_file(f"output/wordclouds/{str(topic_i).rjust(3, '0')}_tfidf.png")


def get_tfidf_topic_weights(corpus_tfidf, W):

    corpus_top_topics = np.argmax(W, axis=1)
    topic_tfidf_weights = []

    for topic_i in range(len(corpus_top_topics)):
        topic_corpus = corpus_tfidf[corpus_top_topics == topic_i]

        topic_word_scores = topic_corpus.sum(axis=0).A1
        topic_tfidf_weights.append(topic_word_scores)

    return np.array(topic_tfidf_weights)
