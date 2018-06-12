# Mark Evers
# Created: 4/9/2018
# webapp.py
# Web application

import os, sys
sys.path.append(os.getcwd() + "/src")

from flask import Flask, request, render_template
import json
from pickle_workaround import pickle_load, pickle_dump
import numpy as np
from summarize import summarize_doc_nmf
from corpus import Corpus

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.add_template_global(str, "str")


topics = json.load(open("data/json/topics.json"))
corpus = pickle_load("data/pickles/corpus.pkl")
vectorizer = pickle_load("data/pickles/vectorizer.pkl")
nmf = pickle_load("data/pickles/nmf.pkl")
W = pickle_load("data/pickles/W.pkl")
W_normalized = W / W.max(axis=0)


@app.route("/topic.html", methods=["GET"])
def index():

    topic_i = int(request.args["topic"]) if "topic" in request.args else 0
    topic_i_str = str(topic_i)
    topic_label = topics[topic_i_str]["label"]
    topic_threshold = topics[topic_i_str]["threshold"]

    topic_docs = W[:, topic_i]
    topic_docs_normalized = W_normalized[:, topic_i]

    top_10_docs_i = np.argsort(topic_docs)[::-1][:10]
    top_10_docs = corpus.get_docs(top_10_docs_i)
    top_10_doc_ids = corpus.doc_ids[top_10_docs_i]
    top_10_docs_nmf_summaries = [summarize_doc_nmf(doc, vectorizer, nmf, topic_i, 2) for doc in top_10_docs]


    return render_template("topic.html", topic_i=topic_i, topic=topics[topic_i_str], summaries=zip(top_10_doc_ids, top_10_docs_nmf_summaries))



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
