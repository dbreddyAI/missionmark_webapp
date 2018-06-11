# Mark Evers
# 6/10/18


from database import get_connection
import globals as g
import pandas as pd
import vectorizers
import numpy as np



class Corpus():

    def __init__(self):

        self._tfidf_corpus = None
        self._vectorizer = None
        self._n_docs = 0
        self._doc_ids = []



    def get_from_query(self, query, count_query):
        conn = get_connection()

        g.debug("Loading corpus...")
        with conn.cursor(name="corpus_getter") as cursor:
            cursor.itersize = g.DOC_BUFFER_SIZE

            cursor.execute(count_query)
            self._n_docs = cursor.fetchone()[0]

            cursor.execute(query)
            for doc_id, doc in cursor:
                self._doc_ids.append(doc_id)
                yield doc



    def create_from_vectorizer(self, vectorizer, query, count_query):

        self._vectorizer = vectorizer
        tfidf_corpus = self._vectorizer.transform(self.get_from_query(query, count_query))
        self._tfidf_corpus = pd.SparseDataFrame(data=tfidf_corpus, index=self._doc_ids, columns=vectorizer.get_feature_names())


    def create(self, query, count_query):

        self._vectorizer = vectorizers.TfidfVectorizerProgressBar(max_features=g.MAX_FEATURES, min_df=g.MIN_DF, max_df=g.MAX_DF, stop_words=vectorizers.get_stopwords(), tokenizer=vectorizers.tokenize, ngram_range=(1, g.N_GRAMS), strip_accents="ascii", sublinear_tf=True, dtype=np.uint16, progress_bar_clear_when_done=True)




def main():

    query = """
                SELECT opportunity_id, program_description
                FROM import.govwin_opportunity
                WHERE program_description ILIKE('%requirements%')
            """
    count_query = """
                      SELECT COUNT(*)
                      FROM import.govwin_opportunity
                      WHERE program_description ILIKE('%requirements%')
                  """

    corpus = Corpus()
    corpus.create_from_vectorizer()