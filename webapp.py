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
import globals as g
import re
regex = re.compile(r"(BACKGROUND|Background\:)[\s\S]+(?=(Requirements\:|REQUIREMENTS|Summary\:|SUMMARY))")


app = Flask(__name__)
app.secret_key = os.urandom(24)
app.add_template_global(str, "str")
app.add_template_global(int, "int")
app.add_template_global(round, "round")




topics = json.load(open("data/json/topics.json"))
corpus = pickle_load("data/pickles/corpus.pkl")
vectorizer = pickle_load("data/pickles/vectorizer.pkl")
nmf = pickle_load("data/pickles/nmf.pkl")
W = pickle_load("data/pickles/W.pkl")
W_normalized = W / W.max(axis=0)



def naics_descriptions(doc_ids):

    from database import get_connection
    query = f"""
                        SELECT opportunity_id, naics_description
                        FROM import.govwin_opportunity
                        WHERE opportunity_id IN ({", ".join([str(x) for x in doc_ids])})
                    """

    conn = get_connection()
    with conn.cursor(name="doc_getter") as cursor:
        cursor.itersize = g.DOC_BUFFER_SIZE
        cursor.execute(query)

        ids = []
        docs = []

        for id, doc in cursor:
            ids.append(id)
            docs.append(doc)

    # return the docs in the same order they were requested
    result = np.array([docs[ids.index(doc_id)] for doc_id in doc_ids])
    return result




@app.route("/", methods=["GET"])
@app.route("/index.html", methods=["GET"])
def index():

    return render_template("index.html", topics=topics)



@app.route("/topic.html", methods=["GET"])
def topic():

    topic_i_str = request.args["topic"] if "topic" in request.args else 0
    topic_i = int(topic_i_str)
    topic_label = topics[topic_i_str]["label"]
    topic_threshold = topics[topic_i_str]["threshold"]

    topic_docs = W[:, topic_i]
    topic_docs_normalized = W_normalized[:, topic_i]

    top_10_docs_i = np.argsort(topic_docs)[::-1][:10]
    # top_10_docs = [doc.replace("\n", "<br />") for doc in corpus.get_docs(top_10_docs_i)]
    top_10_docs = [regex.sub(" ", doc) for doc in corpus.get_docs(top_10_docs_i)]
    top_10_doc_ids = corpus.doc_ids[top_10_docs_i]
    top_10_docs_nmf_summaries = [summarize_doc_nmf(doc, vectorizer, nmf, topic_i, 5) for doc in top_10_docs]
    top_10_docs_descriptions = naics_descriptions(top_10_doc_ids)


    return render_template("topic.html", topic_i=topic_i, topic=topics[topic_i_str], summaries=zip(top_10_doc_ids, top_10_docs_descriptions, top_10_docs_nmf_summaries))



@app.route("/document.html", methods=["GET"])
def document():

    doc_id = int(request.args["doc_id"]) if "doc_id" in request.args else 0
    doc_i = np.where(corpus.doc_ids == doc_id)[0]
    doc = corpus.get_doc_by_id(doc_id)
    doc_topics = W_normalized[doc_i][0] * 100
    alphas = doc_topics / doc_topics.max()
    naics_description = naics_descriptions([doc_id])[0]

    return render_template("document.html", doc_id=doc_id, doc=doc, doc_topics=doc_topics, alphas=alphas, naics_description=naics_description, topics=topics)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
