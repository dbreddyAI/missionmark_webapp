# Mark Evers
# 5/18/18

import numpy
from sys import stdout
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from time import time

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem.porter import PorterStemmer as Stemmer



class ProgressBarVectorizer():
    """
        Base class containing the progress bar functions.  Inherited by child
        classes in conjunction with the scikit-learn classes.

        Parameters
        ----------
        resolution_seconds : float
            How often (in seconds) to write the progress bar to stdout.
        clear_when_done : bool
            Whether or not to clear the progress bar from stdout when it is finished.
    """

    def __init__(self, resolution_seconds, clear_when_done):

        self._n_docs = 0
        self._completed_docs = 0
        self._last_time = 0
        self._resolution_seconds = resolution_seconds
        self._clear_when_done = clear_when_done


    def _analyzer_with_progress_bar(self, analyzer, doc):
        result = analyzer(doc)
        self._completed_docs += 1
        self._progress_bar()
        return result


    def _start_progress(self, n_docs):
        self._n_docs = n_docs
        self._completed_docs = 0
        self._progress_bar()


    def _progress_bar(self):

        if not self._n_docs:
            return

        time_now = time()
        if time_now - self._last_time < self._resolution_seconds and self._completed_docs < self._n_docs:
            return

        # percentage done
        percentage = int(self._completed_docs / self._n_docs * 100)

        stdout.write('\r')
        # print the progress bar
        stdout.write("[{}]{}%".format(("-" * int(percentage / 2) + (">" if percentage < 100 else "")).ljust(50), str(percentage).rjust(4)))
        # print the text figures
        stdout.write(" ({}/{})".format(self._completed_docs, self._n_docs).rjust(14))

        # clear the bar or move on to the next line if we're done
        if self._completed_docs == self._n_docs:
            if self._clear_when_done:
                stdout.write("\r")
                stdout.write(" " * 80)
                stdout.write("\r")
            else:
                stdout.write("\n")

        stdout.flush()

        self._last_time = time_now



class CountVectorizerProgressBar(CountVectorizer, ProgressBarVectorizer):
    """
        scikit-learn's CountVectorizer object, but it displays a progress bar when fitting/transforming!
        Author : Mark Evers
        Github : http://github.com/mevers303

        Parameters
        ----------
        progress_bar_resolution_seconds : float
            How long (in seconds) it should wait in between updating the progress bar.
        progress_bar_clear_when_done : booloan
            Whether or not to clear the progress bar displaying 100% from stdout when it
            is done.


        ORIGINAL DOCSTRING:
        Convert a collection of text documents to a matrix of token counts
        This implementation produces a sparse representation of the counts using
        scipy.sparse.csr_matrix.
        If you do not provide an a-priori dictionary and you do not use an analyzer
        that does some kind of feature selection then the number of features will
        be equal to the vocabulary size found by analyzing the data.
        Read more in the :ref:`User Guide <text_feature_extraction>`.
        Parameters
        ----------
        input : string {'filename', 'file', 'content'}
            If 'filename', the sequence passed as an argument to fit is
            expected to be a list of filenames that need reading to fetch
            the raw content to analyze.
            If 'file', the sequence items must have a 'read' method (file-like
            object) that is called to fetch the bytes in memory.
            Otherwise the input is expected to be the sequence strings or
            bytes items are expected to be analyzed directly.
        encoding : string, 'utf-8' by default.
            If bytes or files are given to analyze, this encoding is used to
            decode.
        decode_error : {'strict', 'ignore', 'replace'}
            Instruction on what to do if a byte sequence is given to analyze that
            contains characters not of the given `encoding`. By default, it is
            'strict', meaning that a UnicodeDecodeError will be raised. Other
            values are 'ignore' and 'replace'.
        strip_accents : {'ascii', 'unicode', None}
            Remove accents during the preprocessing step.
            'ascii' is a fast method that only works on characters that have
            an direct ASCII mapping.
            'unicode' is a slightly slower method that works on any characters.
            None (default) does nothing.
        analyzer : string, {'word', 'char', 'char_wb'} or callable
            Whether the feature should be made of word or character n-grams.
            Option 'char_wb' creates character n-grams only from text inside
            word boundaries; n-grams at the edges of words are padded with space.
            If a callable is passed it is used to extract the sequence of features
            out of the raw, unprocessed input.
        preprocessor : callable or None (default)
            Override the preprocessing (string transformation) stage while
            preserving the tokenizing and n-grams generation steps.
        tokenizer : callable or None (default)
            Override the string tokenization step while preserving the
            preprocessing and n-grams generation steps.
            Only applies if ``analyzer == 'word'``.
        ngram_range : tuple (min_n, max_n)
            The lower and upper boundary of the range of n-values for different
            n-grams to be extracted. All values of n such that min_n <= n <= max_n
            will be used.
        stop_words : string {'english'}, list, or None (default)
            If 'english', a built-in stop word list for English is used.
            If a list, that list is assumed to contain stop words, all of which
            will be removed from the resulting tokens.
            Only applies if ``analyzer == 'word'``.
            If None, no stop words will be used. max_df can be set to a value
            in the range [0.7, 1.0) to automatically detect and filter stop
            words based on intra corpus document frequency of terms.
        lowercase : boolean, True by default
            Convert all characters to lowercase before tokenizing.
        token_pattern : string
            Regular expression denoting what constitutes a "token", only used
            if ``analyzer == 'word'``. The default regexp select tokens of 2
            or more alphanumeric characters (punctuation is completely ignored
            and always treated as a token separator).
        max_df : float in range [0.0, 1.0] or int, default=1.0
            When building the vocabulary ignore terms that have a document
            frequency strictly higher than the given threshold (corpus-specific
            stop words).
            If float, the parameter represents a proportion of documents, integer
            absolute counts.
            This parameter is ignored if vocabulary is not None.
        min_df : float in range [0.0, 1.0] or int, default=1
            When building the vocabulary ignore terms that have a document
            frequency strictly lower than the given threshold. This value is also
            called cut-off in the literature.
            If float, the parameter represents a proportion of documents, integer
            absolute counts.
            This parameter is ignored if vocabulary is not None.
        max_features : int or None, default=None
            If not None, build a vocabulary that only consider the top
            max_features ordered by term frequency across the corpus.
            This parameter is ignored if vocabulary is not None.
        vocabulary : Mapping or iterable, optional
            Either a Mapping (e.g., a dict) where keys are terms and values are
            indices in the feature matrix, or an iterable over terms. If not
            given, a vocabulary is determined from the input documents. Indices
            in the mapping should not be repeated and should not have any gap
            between 0 and the largest index.
        binary : boolean, default=False
            If True, all non zero counts are set to 1. This is useful for discrete
            probabilistic models that model binary events rather than integer
            counts.
        dtype : type, optional
            Type of the matrix returned by fit_transform() or transform().
        Attributes
        ----------
        vocabulary_ : dict
            A mapping of terms to feature indices.
        stop_words_ : set
            Terms that were ignored because they either:
              - occurred in too many documents (`max_df`)
              - occurred in too few documents (`min_df`)
              - were cut off by feature selection (`max_features`).
            This is only available if no vocabulary was given.
        See also
        --------
        HashingVectorizer, TfidfVectorizer
        Notes
        -----
        The ``stop_words_`` attribute can get large and increase the model size
        when pickling. This attribute is provided only for introspection and can
        be safely removed using delattr or set to None before pickling.
        """


    def __init__(self, input="content", encoding="utf-8", decode_error="strict", strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, stop_words=None, token_pattern="(?u)\b\w\w+\b", ngram_range=(1, 1),
                 analyzer="word", max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False,
                 dtype=numpy.int64, progress_bar_resolution_seconds=.333, progress_bar_clear_when_done=False):

        CountVectorizer.__init__(self, input=input, encoding=encoding, decode_error=decode_error, strip_accents=strip_accents,
                                 lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer, stop_words=stop_words,
                                 token_pattern=token_pattern, ngram_range=ngram_range, analyzer=analyzer, max_df=max_df,
                                 min_df=min_df, max_features=max_features, vocabulary=vocabulary, binary=binary, dtype=dtype)

        ProgressBarVectorizer.__init__(self, progress_bar_resolution_seconds, progress_bar_clear_when_done)


    def build_analyzer(self):
        analyzer = super().build_analyzer()
        return lambda doc: self._analyzer_with_progress_bar(analyzer, doc)


    def fit(self, raw_documents, y=None, n_docs=0):
        n = n_docs if n_docs else len(raw_documents)
        self._start_progress(n)
        return super().fit(raw_documents, y=y)


    def fit_transform(self, raw_documents, y=None, n_docs=0):
        n = n_docs if n_docs else len(raw_documents)
        self._start_progress(n)
        return super().fit_transform(raw_documents, y=y)


    def transform(self, raw_documents, n_docs=0):
        n = n_docs if n_docs else len(raw_documents)
        self._start_progress(n)
        return super().transform(raw_documents)



class TfidfVectorizerProgressBar(TfidfVectorizer, ProgressBarVectorizer):
    """
        scikit-learn's TfidfVectorizer object, but it displays a progress bar when fitting/transforming!
        Author : Mark Evers
        Github : http://github.com/mevers303

        Parameters
        ----------
        progress_bar_resolution_seconds : float
            How long (in seconds) it should wait in between updating the progress bar.
        progress_bar_clear_when_done : booloan
            Whether or not to clear the progress bar displaying 100% from stdout when it
            is done.


        ORIGINAL DOCSTRING:
        Convert a collection of raw documents to a matrix of TF-IDF features.
        Equivalent to CountVectorizer followed by TfidfTransformer.
        Read more in the :ref:`User Guide <text_feature_extraction>`.
        Parameters
        ----------
        input : string {'filename', 'file', 'content'}
            If 'filename', the sequence passed as an argument to fit is
            expected to be a list of filenames that need reading to fetch
            the raw content to analyze.
            If 'file', the sequence items must have a 'read' method (file-like
            object) that is called to fetch the bytes in memory.
            Otherwise the input is expected to be the sequence strings or
            bytes items are expected to be analyzed directly.
        encoding : string, 'utf-8' by default.
            If bytes or files are given to analyze, this encoding is used to
            decode.
        decode_error : {'strict', 'ignore', 'replace'}
            Instruction on what to do if a byte sequence is given to analyze that
            contains characters not of the given `encoding`. By default, it is
            'strict', meaning that a UnicodeDecodeError will be raised. Other
            values are 'ignore' and 'replace'.
        strip_accents : {'ascii', 'unicode', None}
            Remove accents during the preprocessing step.
            'ascii' is a fast method that only works on characters that have
            an direct ASCII mapping.
            'unicode' is a slightly slower method that works on any characters.
            None (default) does nothing.
        analyzer : string, {'word', 'char'} or callable
            Whether the feature should be made of word or character n-grams.
            If a callable is passed it is used to extract the sequence of features
            out of the raw, unprocessed input.
        preprocessor : callable or None (default)
            Override the preprocessing (string transformation) stage while
            preserving the tokenizing and n-grams generation steps.
        tokenizer : callable or None (default)
            Override the string tokenization step while preserving the
            preprocessing and n-grams generation steps.
            Only applies if ``analyzer == 'word'``.
        ngram_range : tuple (min_n, max_n)
            The lower and upper boundary of the range of n-values for different
            n-grams to be extracted. All values of n such that min_n <= n <= max_n
            will be used.
        stop_words : string {'english'}, list, or None (default)
            If a string, it is passed to _check_stop_list and the appropriate stop
            list is returned. 'english' is currently the only supported string
            value.
            If a list, that list is assumed to contain stop words, all of which
            will be removed from the resulting tokens.
            Only applies if ``analyzer == 'word'``.
            If None, no stop words will be used. max_df can be set to a value
            in the range [0.7, 1.0) to automatically detect and filter stop
            words based on intra corpus document frequency of terms.
        lowercase : boolean, default True
            Convert all characters to lowercase before tokenizing.
        token_pattern : string
            Regular expression denoting what constitutes a "token", only used
            if ``analyzer == 'word'``. The default regexp selects tokens of 2
            or more alphanumeric characters (punctuation is completely ignored
            and always treated as a token separator).
        max_df : float in range [0.0, 1.0] or int, default=1.0
            When building the vocabulary ignore terms that have a document
            frequency strictly higher than the given threshold (corpus-specific
            stop words).
            If float, the parameter represents a proportion of documents, integer
            absolute counts.
            This parameter is ignored if vocabulary is not None.
        min_df : float in range [0.0, 1.0] or int, default=1
            When building the vocabulary ignore terms that have a document
            frequency strictly lower than the given threshold. This value is also
            called cut-off in the literature.
            If float, the parameter represents a proportion of documents, integer
            absolute counts.
            This parameter is ignored if vocabulary is not None.
        max_features : int or None, default=None
            If not None, build a vocabulary that only consider the top
            max_features ordered by term frequency across the corpus.
            This parameter is ignored if vocabulary is not None.
        vocabulary : Mapping or iterable, optional
            Either a Mapping (e.g., a dict) where keys are terms and values are
            indices in the feature matrix, or an iterable over terms. If not
            given, a vocabulary is determined from the input documents.
        binary : boolean, default=False
            If True, all non-zero term counts are set to 1. This does not mean
            outputs will have only 0/1 values, only that the tf term in tf-idf
            is binary. (Set idf and normalization to False to get 0/1 outputs.)
        dtype : type, optional
            Type of the matrix returned by fit_transform() or transform().
        norm : 'l1', 'l2' or None, optional
            Norm used to normalize term vectors. None for no normalization.
        use_idf : boolean, default=True
            Enable inverse-document-frequency reweighting.
        smooth_idf : boolean, default=True
            Smooth idf weights by adding one to document frequencies, as if an
            extra document was seen containing every term in the collection
            exactly once. Prevents zero divisions.
        sublinear_tf : boolean, default=False
            Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
        Attributes
        ----------
        vocabulary_ : dict
            A mapping of terms to feature indices.
        idf_ : array, shape = [n_features], or None
            The learned idf vector (global term weights)
            when ``use_idf`` is set to True, None otherwise.
        stop_words_ : set
            Terms that were ignored because they either:
              - occurred in too many documents (`max_df`)
              - occurred in too few documents (`min_df`)
              - were cut off by feature selection (`max_features`).
            This is only available if no vocabulary was given.
        See also
        --------
        CountVectorizer
            Tokenize the documents and count the occurrences of token and return
            them as a sparse matrix
        TfidfTransformer
            Apply Term Frequency Inverse Document Frequency normalization to a
            sparse matrix of occurrence counts.
        Notes
        -----
        The ``stop_words_`` attribute can get large and increase the model size
        when pickling. This attribute is provided only for introspection and can
        be safely removed using delattr or set to None before pickling.
        """


    def __init__(self, input="content", encoding="utf-8", decode_error="strict", strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer="word", stop_words=None, token_pattern="(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False,
                 dtype=numpy.int64, norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False,
                 progress_bar_resolution_seconds=.333, progress_bar_clear_when_done=False):

        TfidfVectorizer.__init__(self, input=input, encoding=encoding, decode_error=decode_error, strip_accents=strip_accents,
                         lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer, stop_words=stop_words,
                         token_pattern=token_pattern, ngram_range=ngram_range, analyzer=analyzer, max_df=max_df,
                         min_df=min_df, max_features=max_features, vocabulary=vocabulary, binary=binary, dtype=dtype,
                         norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)

        ProgressBarVectorizer.__init__(self, progress_bar_resolution_seconds, progress_bar_clear_when_done)


    def build_analyzer(self):
        analyzer = super().build_analyzer()
        return lambda doc: self._analyzer_with_progress_bar(analyzer, doc)


    def fit(self, raw_documents, y=None, n_docs=0):
        n = n_docs if n_docs else len(raw_documents)
        self._start_progress(n)
        return super().fit(raw_documents, y=y)


    def fit_transform(self, raw_documents, y=None, n_docs=0):
        n = n_docs if n_docs else len(raw_documents)
        self._start_progress(n)
        return super().fit_transform(raw_documents, y=y)


    def transform(self, raw_documents, copy=True, n_docs=0):
        n = n_docs if n_docs else len(raw_documents)
        self._start_progress(n)
        return super().transform(raw_documents, copy=copy)



def get_stopwords():
    """
    Reads the stopwords from stopwords_raw.txt and combines them with the ENGLISH_STOP_WORDS in sklearn.
    :return: A set of stopwords.
    """

    with open("stopwords/stopwords_tokenize.txt", "r") as f:
        custom_stopwords = set(tokenize(f.read().lower()))

    return list(ENGLISH_STOP_WORDS.union(custom_stopwords))


def get_stopphrases():

    with open("stopwords/stopwords_phrases.txt", "r") as f:
        stop_phrases = set([re.escape(line.lower().strip()) for line in f if len(line) > 3])

    with open("stopwords/stopwords_regex.txt", "r") as f:
        stop_phrases.update([line.lower().strip() for line in f if len(line) > 3])

    return re.compile("|".join(stop_phrases), re.DOTALL)


_regex_split = re.compile(r"[^a-z]+")
def tokenize(text):
    """
    Tokenizes a document to be converted to a TF-IDF vector.
    :param text: The text/document to be tokenized.
    :return: A list of stemmed tokens.
    """
    global _stemmer, _stop_phrases
    text = _stop_phrases.sub(" ", text)
    result = [_stemmer.stem(token) for token in _regex_split.split(text) if len(token) > 3]
    return result


_stop_phrases = get_stopphrases()
_stemmer = Stemmer()
