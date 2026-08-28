import networkx as nx
import pandas as pd
import numpy as np
import streamlit as st
import time
from gensim.models import Word2Vec
from typing import Literal
from enum import Enum

# create a wrapper class for a nx.DiGraph to allow for easier use by external functions
# or is that just more confusing for people using this later?
# yeah just return an nx.DiGraph, that would be so much more straightforward
# So vocabnet goes
# Make this more functional than OO - have word_search as a standalone function
# Add co-oc and embedding algorithms here to allow for on-the-fly recalculation
# And on that note, figure out what the practical limits for drawing inferences based
# on dataset size are - there should be papers on that


# This enum allows for setting different algorithms.
class Algorithm(Enum):
    CON = 0 # co-occurrence
    W2V = 1 # Word2Vec

# This enum allows setting different sources.
class Source(Enum):
    H = 0 # Hebrew Old Testament
    G = 1 # Greek New Testament
    E = 2 # Empty
    A = 3 # Aramaic Old Testament
    S = 4 # Septuagint


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


def _initialize_con_df(lemmas: pd.Series) -> pd.DataFrame:
    # Little bit of a conceptual help from ChatGPT on dataframe initialization
    # TODO: Is there a more efficient way to do this?
    df = pd.DataFrame(index=lemmas, columns=lemmas, dtype=int)
    for col in df.columns:
        df[col] = 0
    return df


def _most_similar(algo: Algorithm, word: str, df: pd.DataFrame, topn: int) -> pd.Series:
    match algo:
        case Algorithm.CON:
            return pd.Series(df.drop(word, axis=0)[word].nlargest(topn))
        case Algorithm.W2V:
            return pd.Series(df.drop(word, axis=0)[word].nlargest(topn))

# What is this function doing here??
def process_book_input(book: str) -> str:
    return "00"


class NetBuilder:
    def __init__(self):
        """Loads datasets from the resources directory and creates an empty DiGraph,
        storing them in a new NetBuilder object."""
        # To do - make this a dictionary of datasets eventually.
        self.hb: pd.DataFrame = pd.read_parquet("resources/hb.parquet")
        self.gnt: pd.DataFrame = pd.read_parquet("resources/gnt.parquet")
        self.dg: nx.DiGraph = nx.DiGraph()

    def lex_to_strongs(self, source: Source, lex: str) -> list[tuple[Source, int]]:
        # TODO: Stop trying to make these all one-liners.
        try:
            match source:
                case source.H:
                    return [[(Source[i[0]], int(i[1:].strip())) for i in item.split("＋")] for item in self.hb['strongno'][self.hb['lemma'] == lex].unique()][0]
                case source.G:
                    return [[(Source[i[0]], int(i[1:].strip())) for i in item.split("＋")] for item in self.gnt['strongno'][self.gnt['lemma'] == lex].unique()][0]
        except:
            return [(Source.E, 0)]

    def translit_to_raw(self, source: Source, lex: str) -> str:
        raw_lemma = ""
        try:
            if source == Source.H:
                raw_lemma = self.hb.loc[self.hb['lemma'] == lex]['display_lemma'].iloc[0]
            elif source == Source.G:
                raw_lemma = self.gnt.loc[self.gnt['lemma'] == lex]['display_lemma'].iloc[0]
            else:
                raise NotImplementedError
        except IndexError:
            print(f"index error transliterating {lex}")
            raise IndexError
        return raw_lemma
    
    def fetch_gloss(self, source: Source, lex: str) -> str:
        # TODO: would it be possible to combine these data fetching calls into a single function with varying parameters? Better API design than just ad-hoc
        gloss = ""
        if source == Source.H:
            gloss = self.hb.loc[self.hb['lemma'] == lex]['gloss'].iloc[0]
        elif source == Source.G:
            gloss = self.gnt.loc[self.gnt['lemma'] == lex]['gloss'].iloc[0]
        else:
            raise NotImplementedError
        return gloss

    def generate_comat(self, source: Source, window_size = 3, included_books = None) -> pd.DataFrame:
        t10 = time.perf_counter()
        df = self.hb if source == Source.H else self.gnt
    

        if included_books:
            df = df[df['book'].astype(int).isin(included_books)]
            df = df.reset_index(drop=True)
            print(f"incl_books, {included_books}, dflen {len(df)}, df {df}")

        t11 = time.perf_counter()
        print(f"Filtered df in {t11 - t10:.4f} seconds")
        t12 = time.perf_counter()
        wordmap: dict = {w: i for i, w in enumerate(df['lemma'])}
        lemmas: pd.Series = pd.Series(wordmap.keys(), index = list(range(len(wordmap))) ) 
        lemma_index = pd.Index(lemmas)

        word_count = len(df['lemma'])
        lemmalen = len(lemmas)
        t13 = time.perf_counter()
        print(f"Generated lemma index in {t13 - t12:.4f} seconds")
        
        t14 = time.perf_counter()
        # performance draw is here
        comat = np.zeros((lemmalen,lemmalen), dtype=int)
        i1_total = 0
        i2_total = 0
        i3_total = 0
        for idx, lemma in enumerate(df['lemma']):
            for j in range(max(0, idx - window_size), min(word_count, idx + window_size + 1)):
                t15 = time.perf_counter()
                idx0 = lemma_index.get_loc(lemma)
                t16 = time.perf_counter()
                idx1 = lemma_index.get_loc(df['lemma'][j]) # so much time is getting spent here (64% of the time.)
                t17 = time.perf_counter()
                comat[idx0][idx1] += 1
                t18 = time.perf_counter()
                i1_total += t16-t15
                i2_total += t17-t16
                i3_total += t18-t17
        print(f"i1_total: {i1_total:.4f}, i2_total: {i2_total:.4f}, i3_total: {i3_total:.4f}")
            
        t15 = time.perf_counter()
        print(f"Generated co-occurrence matrix in {t15 - t14:.4f} seconds")
        return pd.DataFrame(comat, index=lemmas, columns=lemmas)

    def process_strongs_input(self, code: Literal['H', 'G'], num: int) -> str:
        '''TODO: Add param docs here'''
        # Do this with dataclasses; this annoys me
        if code == 'H':
            return self.hb.loc[self.hb['strongno']==f"H{num}"]['lemma'].iloc[0]
        elif code == 'G':
            return self.gnt.loc[self.gnt['strongno']==f"G{num}"]['lemma'].iloc[0]
        else:
            raise ValueError("Incorrect Strong Number")
        
    def _retrain_w2v_model(self, source: Source):
        df = self.hb if source == Source.H else self.gnt
        model = Word2Vec(sentences=Corpus(df))
        wvlen = len(model.wv)
        arr = np.zeros((wvlen, wvlen), dtype=float)
        for i in range(wvlen):
            if i % 50 == 0:
                print(f"Calculating similarity for index {i}")
            for j in range(wvlen):
                arr[i][j] = model.wv.similarity(i, j)
        assert(np.array_equal(arr, arr.T)) # make sure matrix is symmetric
        data = pd.DataFrame(arr, index = model.wv.index_to_key, columns = model.wv.index_to_key)
        data.to_parquet(f"resources/{source.name.lower()}_w2v.parquet")
        return data
        
    def _generate_w2v_similarity_matrix(self, source: Source, retrain: bool = False) -> pd.DataFrame:
        if retrain:
            return self._retrain_w2v_model(source)
        else:
            return pd.read_parquet(f"resources/{source.name.lower()}_w2v.parquet")
            
        
    def _build_word_search_network(self,
                                df: pd.DataFrame,
                                algo: Algorithm,
                                search_word: str,
                                num_steps: int,
                                words_per_level: int,
                                source: Source,
                                words_to_exclude: list[str] | None = None,
                                first: bool = False
                                ):
        # Make sure the graph doesn't have stuff already in it
        # print(f"calling with num_steps = {num_steps}, word = {search_word}")
#         # Initialize an empty list passed to recursive calls to avoid readding words
        if words_to_exclude is None:
            words_to_exclude = [search_word]
        if num_steps > 0:
            t8 = time.perf_counter()
            # ~53% of the time in this algorithm is spent in this line of code \/
            most_similar: pd.Series = _most_similar(algo, search_word, df, words_per_level)
            t9 = time.perf_counter()
            print(f"Calculated most similar words to {search_word} in {t9 - t8:.4f} seconds")
            # print(f"Most similar to {search_word}: \n{most_similar}")
            if first:
                self.dg.add_node(search_word, tag="root")
            for rel_word, similarity in zip(most_similar.index, most_similar):
                t4 = time.perf_counter()
                # ~49% of the time in this algorithm is spent in this line of code \/
                if ((source == Source.H) and self.hb.loc[self.hb['lemma'] == rel_word].empty) or ((source == Source.G) and
                                                                                                  self.gnt.loc[self.gnt['lemma'] == rel_word].empty):
                    words_to_exclude.append(rel_word)
                    continue
                t5 = time.perf_counter()
                # print(f"Checked for existence of {rel_word} in {source.name} in {t5 - t4:.4f} seconds")
                t6 = time.perf_counter()
                self.dg.add_weighted_edges_from([(search_word, rel_word, similarity)]) #3.59e-4% of total time
                t7 = time.perf_counter()
                # print(f"Added edge from {search_word} to {rel_word} in {t7 - t6:.10f} seconds")
                if rel_word not in words_to_exclude:
                    words_to_exclude.append(rel_word)
                    self._build_word_search_network(df,
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
                                    books_to_include: list[str],
                                    **kwargs
                                    ):
        t0 = time.perf_counter()
        df = None

        retrain = False
        if 'retrain' in kwargs:
            retrain = kwargs['retrain']

        word = _self.process_strongs_input(unparsed_word[0], int(unparsed_word[1:])) #type: ignore
        source = Source[unparsed_word[0]]

        t1 = time.perf_counter()
        print(f"Processed word {unparsed_word} to {word} in {t1 - t0:.4f} seconds") # autogen
        if algo == Algorithm.CON:
            df = _self.generate_comat(source, included_books=books_to_include)            
        elif algo == Algorithm.W2V:
            df = _self._generate_w2v_similarity_matrix(source=source, retrain=retrain) 
        else:
            raise NotImplementedError
        t2 = time.perf_counter()
        print(f"Generated similarity matrix in {t2 - t1:.4f} seconds") # autogen
        assert(df is not None)
        print(df)
        _self._build_word_search_network(df,
                                        algo,
                                        word,
                                        num_steps,
                                        words_per_level,
                                        source,
                                        words_to_exclude=None,
                                        first = True
                                        )
        t3 = time.perf_counter()
        print(f"Built network in {t3 - t2:.4f} seconds")


    def get_network(self) -> nx.DiGraph:
        return self.dg
