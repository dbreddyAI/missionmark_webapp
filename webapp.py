# Mark Evers
# Created: 4/9/2018
# webapp.py
# Web application


import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, send_from_directory, request, flash

from sklearn.decomposition import NMF

from src.data import get_query_corpus
from src.pickle_workaround import pickle_load
from vectorizer import tokenize
from src.TfidfVectorizer import TfidfVectorizer
from summaries import summarize_doc



# these are global variables
query = open("output/last_query.txt", "r").read()
strip_html = True
corpus_df = pd.DataFrame()
doc_topic_df = np.array([])
percentages = "topic"
W = pickle_load("output/W.pkl")
W_max = W.max(axis=0)


nmf = pickle_load("output/NMF.pkl")
tfidf = TfidfVectorizer(pickle_load("output/CountVectorizer.pkl"), pickle_load("output/TfidfTransformer.pkl"))


def process_query(_query, _strip_html):

    doc_ids, corpus = get_query_corpus(_query, _strip_html)
    _corpus_df = pd.DataFrame(corpus, index=doc_ids, columns=["text"])
    tfidf_corpus = tfidf.transform(corpus)
    nmf_corpus = nmf.transform(tfidf_corpus)

    return _corpus_df, nmf_corpus







app = Flask(__name__)
app.secret_key = os.urandom(24)
app.add_template_global(zip, "zip")


@app.route("/index.html", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def index():
    global query, strip_html, corpus_df, doc_topic_df, percentages, W_max

    if "query" in request.form:
        query = request.form["query"]
        open("output/last_query.txt", "w").write(query)  # save the last query
        strip_html = "strip_html" in request.form

        corpus_df, doc_topic_df = process_query(query, strip_html)
        doc_topic_df = doc_topic_df / W_max
        display_results = doc_topic_df
    else:
        display_results = doc_topic_df

    if "percentages" in request.args:
        percentages = request.args["percentages"]
    if percentages == "doc":
        display_results = doc_topic_df / doc_topic_df.sum(axis=1).reshape(-1, 1)

    return render_template("index.html", query=query, strip_html=strip_html, doc_ids=corpus_df.index, percentages=percentages, results=display_results)



@app.route("/document.html", methods=["GET"])
def display_doc():

    global corpus_df, W_max

    if "doc_id" in request.args:
        doc_id = request.args["doc_id"]
        if doc_id.isdigit():
            doc_id = int(doc_id)
    else:
        doc_id = corpus_df.index[0]

    doc = corpus_df.loc[doc_id].text
    summary = summarize_doc(doc, tfidf, 3)

    doc_tfidf = tfidf.transform([doc])
    topic_result = nmf.transform(doc_tfidf)[0] / W_max
    doc_result = topic_result / topic_result.sum()
    top_topics_i = np.argsort(topic_result)[::-1][:4]

    return render_template("document.html", doc_id=doc_id, doc=doc, summary=summary, topic_result=topic_result, doc_result=doc_result, top_topics=top_topics_i)






if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
