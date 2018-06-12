# Mark Evers
# 6/10/18


from database import get_connection
import globals as g
import vectorizers
import numpy as np
from pickle_workaround import pickle_dump
import text_processing



class Corpus():

    def __init__(self, id_column, text_column, table_name, where_clause, strip_html=True):

        self._tfidf_corpus = None
        self._n_docs = 0
        self._doc_ids = None

        self._id_column = id_column
        self._text_column = text_column
        self._table_name = table_name
        self._where_clause = where_clause
        self._strip_html = strip_html


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

        g.debug(f" -> Found {self._n_docs} documents")
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
                if self._strip_html:
                    doc = text_processing.strip_html(doc)
                yield doc

        self._doc_ids = np.array(doc_ids)


    def create_from_transform(self, vectorizer):

        self.query_n_docs()
        g.debug("Downloading and vectorizing documents...")
        self._tfidf_corpus = vectorizer.transform(self.query(), n_docs=self._n_docs)
        g.debug(" -> Done", 1)

        return self


    def create_new_vectorizer(self):

        self.query_n_docs()
        g.debug("Downloading and vectorizing documents...")
        vectorizer = vectorizers.TfidfVectorizerProgressBar(max_features=g.MAX_FEATURES, min_df=g.MIN_DF, max_df=g.MAX_DF, stop_words=vectorizers.get_stopwords(), tokenizer=vectorizers.tokenize, ngram_range=(1, g.N_GRAMS), strip_accents="ascii", sublinear_tf=True, dtype=np.uint16, progress_bar_clear_when_done=True)
        self._tfidf_corpus = vectorizer.fit_transform(self.query(), n_docs=self._n_docs)
        g.debug(f" -> Found {len(vectorizer.get_feature_names())} features")

        return vectorizer


    def get_doc(self, doc_i):

        g.debug(f"Retrieving doc {doc_id}...", 2)

        doc_id = self._doc_ids[doc_i]
        query = f"""
                    SELECT {self._text_column}
                    FROM {self._table_name}
                    WHERE {self._id_column} = {doc_id}
                """

        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(query)
            doc = cursor.fetchone()[0]
            if self._strip_html:
                doc = text_processing.strip_html(doc)

        return doc


    def get_docs(self, doc_is):
        g.debug(f"Retrieving {len(doc_is)} documents...", 2)

        doc_ids = self._doc_ids[doc_is]
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
                ids.append(id)
                if self._strip_html:
                    doc = text_processing.strip_html(doc)
                docs.append(doc)

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
