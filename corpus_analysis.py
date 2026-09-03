import networkx as nx
import pandas as pd
import numpy as np
import streamlit as st
from gensim.models import Word2Vec
from typing import Literal
from enum import Enum


# This enum allows for setting different algorithms.
class Algorithm(Enum):
    CON = 0 # co-occurrence
    W2V = 1 # Word2Vec

# This enum allows setting different sources.
class Source(Enum):
    HBB = 0 # Hebrew Bible
    GNT = 1 # Greek New Testament
    NUL = 2 # Empty
    AOT = 3 # Aramaic Old Testament, currently not implemented
    LXX = 4 # Septuagint, currently not implemented

# Maps Strong's prefix letters to sources. Will be replaced with a different corpus-based system when AOT and LXX are added.
strongs_to_source = {'H': Source.HBB, 'G': Source.GNT}

# The corpus class required for Gensim training.
class Corpus:
    def __init__(self, df):
        self.df = df
    
    def __iter__(self):
        """Returns a generator that returns a list of each book in turn."""
        if self.df is not None:
            grouped = self.df.groupby('book')
            for _, group in grouped:
                yield list(group['lemma'])

# Finds the most similar words in the given dataframe.
def _most_similar(word: str, df: pd.DataFrame, topn: int) -> pd.Series:
    return pd.Series(df.drop(word, axis=0)[word].nlargest(topn))


class NetBuilder:
    def __init__(self):
        """Loads datasets from the resources directory and creates an empty DiGraph,
        storing them in a new NetBuilder object."""
        self.dg: nx.DiGraph = nx.DiGraph()
        # read all Source enum values and create a dictionary mapping them to dataframes
        self.datasets = {}
        for source in Source:
            try:
                self.datasets[source] = pd.read_parquet(f"resources/{source.name.lower()}.parquet")
            except FileNotFoundError:
                print(f"WARNING: No dataset found for {source.name}.")

    def lex_to_strongs(self, source: Source, lex: str) -> list[tuple[Source, int]]:
        """Takes a corpus (the source parameter) and a lexeme (the lex parameter) and returns a list of Source-int tuples representing
         the possible IDs of the lexeme (there may be more than one).
         Returns (Source.NUL, 0) if an exception is thrown."""
        try:
            return [[(strongs_to_source[i[0]], int(i[1:].strip())) for i in item.split("＋")] for item in self.datasets[source]['strongno'][self.datasets[source]['lemma'] == lex].unique()][0]
        except Exception as e:
            print("ERROR:", e)
            return [(Source.NUL, 0)]

    def translit_to_raw(self, source: Source, lex: str) -> str:
        """Takes a corpus (the source parameter) and a transliterated lexeme (the lex parameter) and returns the lexeme in its original script."""
        try:
            return self.datasets[source].loc[self.datasets[source]['lemma'] == lex]['display_lemma'].iloc[0]
        except IndexError as e:
            print(f"ERROR: Index error transliterating {lex}.", e)
            raise IndexError
    
    def fetch_gloss(self, source: Source, lex: str) -> str:
        """Takes a corpus (the source parameter) and a transliterated lexeme (the lex parameter) and returns the gloss for that lexeme."""
        return self.datasets[source].loc[self.datasets[source]['lemma'] == lex]['gloss'].iloc[0]

    def _generate_comat(self, source: Source, window_size: int = 3, included_books: list | None = None) -> pd.DataFrame:
        """Generates a co-occurrence matrix for the given corpus and parameters.

        source: The Source of the corpus (e.g., Source.HBB)
        window_size: The size of the window to consider for co-occurrence (default is 3; analyzing 3 words before and after each target for a total span of 7 words)
        included_books: A list of book IDs to include in the analysis (default is None, which includes all books)
        """
        # A good deal of credit for this algorithm goes to https://www.geeksforgeeks.org/nlp/co-occurence-matrix-in-nlp/.

        df = self.datasets[source]

        if included_books:
            df = df[df['book'].astype(int).isin(included_books)]
            df = df.reset_index(drop=True)

        wordmap: dict = {w: i for i, w in enumerate(df['lemma'])}
        lemmas: pd.Series = pd.Series(wordmap.keys(), index = list(range(len(wordmap))) ) 
        lemma_index = pd.Index(lemmas)

        word_count = len(df['lemma'])
        lemmalen = len(lemmas)

        comat = np.zeros((lemmalen,lemmalen), dtype=int)


        for idx, lemma in enumerate(df['lemma']):
            for j in range(max(0, idx - window_size), min(word_count, idx + window_size + 1)):
                idx0 = lemma_index.get_loc(lemma)
                idx1 = lemma_index.get_loc(df['lemma'][j])
                comat[idx0][idx1] += 1

        return pd.DataFrame(comat, index=lemmas, columns=lemmas)

    def process_strongs_input(self, code: Literal['H', 'G'], num: int) -> str:
        """Takes a Strong's number as two separate params, 'H' or 'G' plus the number id. Returns the corresponding lexeme."""
        try:
            source = strongs_to_source[code]
            return self.datasets[source].loc[self.datasets[source]['strongno'] == f"{code}{num}"]['lemma'].iloc[0]
        except KeyError:
            raise ValueError(f"Invalid Strong's code: {code}")
        except IndexError:
            raise IndexError(f"Lexeme for {code} not found in the dataset.")

    def _retrain_w2v_model(self, source: Source):
        """Retrains the Word2Vec model for the given corpus and saves the resulting matrix to a parquet file."""
        df = self.datasets[source]
        model = Word2Vec(sentences=Corpus(df))
        wvlen = len(model.wv)
        arr = np.zeros((wvlen, wvlen), dtype=float)
        for i in range(wvlen):
            if i % 100 == 0:
                print(f"Similarity calculation progress: {i} / {wvlen}")
            for j in range(wvlen):
                arr[i][j] = model.wv.similarity(i, j)
        assert(np.array_equal(arr, arr.T)) # make sure matrix is symmetric
        data = pd.DataFrame(arr, index = model.wv.index_to_key, columns = model.wv.index_to_key)
        data.to_parquet(f"resources/{source.name.lower()}_w2v.parquet")
        return data
        
    def _get_w2v_similarity_matrix(self, source: Source, retrain: bool = False) -> pd.DataFrame:
        """Returns the stored W2V data for a given corpus.
        source: determines the corpus to return.
        retrain: if True, the W2V model will be retrained. Should only be set to true when initially training the model."""
        if retrain:
            return self._retrain_w2v_model(source)
        else:
            return pd.read_parquet(f"resources/{source.name.lower()}_w2v.parquet")
            
        
    def _add_words_to_network(self,
                                df: pd.DataFrame,
                                algo: Algorithm,
                                search_word: str,
                                num_steps: int,
                                words_per_level: int,
                                source: Source,
                                words_to_exclude: list[str] | None = None
                                ):
        """Generates and returns a network of lexemes for display. Recursive.
        df: the dataframe of words to analyze for similarity.
        algo:  which algorithm to use for similarity calculation.
        search_word: the specific lemma to start the network from.
        num_steps: controls recursion depth.
        words_per_level: controls how many similar words are returned at each level.
        source: which corpus is being analyzed.
        words_to_exclude: words already added to the network; avoids duplicate nodes.
        """

        if words_to_exclude is None:
            words_to_exclude = [search_word]

        if num_steps > 0:
            most_similar: pd.Series = _most_similar(search_word, df, words_per_level)
            # Loop through all similiar words
            for rel_word, similarity in zip(most_similar.index, most_similar):
                if self.datasets[source].loc[self.datasets[source]['lemma'] == rel_word].empty:
                    words_to_exclude.append(rel_word)
                    continue

                self.dg.add_weighted_edges_from([(search_word, rel_word, similarity)])
                # Add the related word to the network
                if rel_word not in words_to_exclude:
                    words_to_exclude.append(rel_word)
                    self._add_words_to_network(df,
                                                algo,
                                                rel_word,
                                                num_steps-1,
                                                words_per_level,
                                                source,
                                                words_to_exclude
                                                )


    @st.cache_data
    def generate_word_search_network(_self,
                                     algo: Algorithm,
                                     unparsed_word: str,
                                     num_steps: int,
                                     words_per_level: int,
                                     books_to_include: list[str] | None,
                                     **kwargs
                                     ):
        """Main function for building a lexical network.

        _self: renamed from self to prevent Streamlit from trying to hash it.
        algo: which algorithm to use for similarity.
        unparsed_word: an unparsed Strong's code for the target word.
        num_steps: recursion depth.
        words_per_level: search breadth at each level of recursion.
        books_to_include: a list of which specific books to include, or None if all books are desired.

        kwargs
        retrain: retrains the W2V model (only if Algorithm.W2V is passed in the algo parameter).
        Should only be used for initial model training.
        """

        retrain = False
        if 'retrain' in kwargs:
            retrain = kwargs['retrain']

        word = _self.process_strongs_input(unparsed_word[0], int(unparsed_word[1:])) #type: ignore
        source = strongs_to_source[unparsed_word[0]]

        df = None

        if algo == Algorithm.CON:
            df = _self._generate_comat(source, included_books=books_to_include)
        elif algo == Algorithm.W2V:
            df = _self._get_w2v_similarity_matrix(source=source, retrain=retrain)
        else:
            raise NotImplementedError

        assert(df is not None)

        _self.dg.add_node(word, tag="root")

        _self._add_words_to_network(df,
                                     algo,
                                     word,
                                     num_steps,
                                     words_per_level,
                                     source,
                                     words_to_exclude=None
                                     )

    def get_network(self) -> nx.DiGraph:
        return self.dg
