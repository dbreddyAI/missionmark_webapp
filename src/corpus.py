# Mark Evers
# 6/10/18


from database import get_connection
import globals as g
import vectorizers
import numpy as np
from pickle_workaround import pickle_dump



class Corpus():

    def __init__(self, id_column, text_column, table_name, where_clause):

        self._tfidf_corpus = None
        self._n_docs = 0
        self._doc_ids = None

        self._id_column = id_column
        self._text_column = text_column
        self._table_name = table_name
        self._where_clause = where_clause


    @property
    def tfidf_corpus(self):
        return self._tfidf_corpus
    
    @property
    def doc_ids(self):
        return self._doc_ids
    
    @property
    def n_docs(self):
        return self._n_docs

    def __getitem__(self, item):
        return self._tfidf_corpus[item]


    def query_n_docs(self):

        g.debug("Loading corpus...")

        count_query = f"""
                          SELECT COUNT(*)
                          FROM {self._table_name}
                          WHERE {self._where_clause}
                      """

        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(count_query)
            self._n_docs = cursor.fetchone()[0]

        g.debug(f" ->  Found {self._n_docs} documents", 1)
        return self.n_docs


    def query(self):

        # g.debug("Downloading corpus...")

        query = f"""
                    SELECT {self._id_column}, {self._text_column}
                    FROM {self._table_name}
                    WHERE {self._where_clause}
                """

        conn = get_connection()
        with conn.cursor(name="corpus_getter") as cursor:
            cursor.itersize = g.DOC_BUFFER_SIZE
            doc_ids = []

            cursor.execute(query)
            for doc_id, doc in cursor:
                doc_ids.append(doc_id)
                yield doc

        self._doc_ids = np.array(doc_ids)
        g.debug(f" -> {self._n_docs} documents downloaded and vectorized", 1)


    def create_from_transform(self, vectorizer):

        self.query_n_docs()
        self._tfidf_corpus = vectorizer.transform(self.query(), n_docs=self._n_docs)

        return self


    def create_new_vectorizer(self):

        self.query_n_docs()
        vectorizer = vectorizers.TfidfVectorizerProgressBar(max_features=g.MAX_FEATURES, min_df=g.MIN_DF, max_df=g.MAX_DF, stop_words=vectorizers.get_stopwords(), tokenizer=vectorizers.tokenize, ngram_range=(1, g.N_GRAMS), strip_accents="ascii", sublinear_tf=True, dtype=np.uint16, progress_bar_clear_when_done=True)
        self._tfidf_corpus = vectorizer.fit_transform(self.query(), n_docs=self._n_docs)

        return vectorizer


    def get_doc(self, doc_id):

        g.debug(f"Retrieving doc {doc_id}...", 2)

        query = f"""
                    SELECT {self._text_column}
                    FROM {self._table_name}
                    WHERE {self._id_column} = {doc_id}
                """

        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(query)
            doc = cursor.fetchone()[0]

        return doc


    def get_docs(self, doc_ids):
        g.debug(f"Retrieving {len(doc_ids)} documents...", 2)

        query = f"""
                    SELECT {self._id_column}, {self._text_column}
                    FROM {self._table_name}
                    WHERE {self._id_column} IN ({", ".join([str(x) for x in doc_ids])})
                """

        conn = get_connection()
        with conn.cursor(name="doc_getter") as cursor:
            cursor.itersize = g.DOC_BUFFER_SIZE
            cursor.execute(query)

            ids = []
            docs = []

            for id, doc in cursor:
                docs.append(doc)
                ids.append(id)

        # return the docs in the same order they were requested
        result = np.array([docs[ids.index(doc_id)] for doc_id in doc_ids])
        return result


def main():

    corpus = Corpus("opportunity_id", "program_description", "import.govwin_opportunity", "program_description ILIKE('%requirements%')")
    vectorizer = corpus.create_new_vectorizer()

    pickle_dump(corpus, "data/pickles/corpus.pkl")
    pickle_dump(vectorizer, "data/pickles/vectorizer.pkl")


if __name__ == "__main__":
    main()
