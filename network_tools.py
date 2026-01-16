import networkx as nx
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim import utils
from typing import Literal
from enum import Enum
import streamlit as st

# create a wrapper class for a nx.DiGraph to allow for easier use by external functions
# or is that just more confusing for people using this later?
# yeah just return an nx.DiGraph, that would be so much more straightforward
# So vocabnet goes
# Make this more functional than OO - have word_search as a standalone function
# Add co-oc and embedding algorithms here to allow for on-the-fly recalculation
# And on that note, figure out what the practical limits for drawing inferences based
# on dataset size are - there should be papers on that

class Algorithm(Enum):
    CON = 0
    W2V = 1

class Source(Enum):
    H = 0
    G = 1


class Corpus:
    def __init__(self, df):
        self.df = df
    
    def __iter__(self):
        '''Returns a generator that returns a list of each book in turn.'''
        if self.df is not None:
            grouped = self.df.groupby('book')
            for _, group in grouped:
                yield list(group['lemma'])

class NetBuilder():
    def __init__(self):
        self.hb: pd.DataFrame = pd.read_parquet("resources/hb.parquet")
        self.gnt: pd.DataFrame = pd.read_parquet("resources/gnt.parquet")
        self.dg: nx.DiGraph = nx.DiGraph()

    def _initialize_con_df(self, lemmas: pd.Series) -> pd.DataFrame:
        # Little bit of a conceptual help from ChatGPT on dataframe initialization
        # TODO: Is there a more efficient way to do this?
        df = pd.DataFrame(index=lemmas, columns=lemmas, dtype=int)
        for col in df.columns:
            df[col] = 0
        return df

    def _most_similar(self, algo: Algorithm, word: str, df: pd.DataFrame, topn: int) -> pd.Series:
        match algo:
            case Algorithm.CON:
                return pd.Series(df.drop(word, axis=0)[word].nlargest(topn))     
            case Algorithm.W2V:
                return pd.Series(df.drop(word, axis=0)[word].nlargest(topn))
        
    def lex_to_strongs(self, source: Source, lex: str) -> list[tuple[Source, int]]:
        # TODO: Stop trying to make these all one-liners.
        match source:
            case source.H:
                return [[(Source[i[0]], int(i[1:].strip())) for i in item.split("＋")] for item in self.hb['strongno'][self.hb['lemma'] == lex].unique()][0]
            case source.G:
                return [[(Source[i[0]], int(i[1:].strip())) for i in item.split("＋")] for item in self.gnt['strongno'][self.gnt['lemma'] == lex].unique()][0]

    
    def generate_comat(self, source: Source, window_size = 3, included_books = None) -> pd.DataFrame:
        df = self.hb if source == Source.H else self.gnt
        
        if included_books:
            df = df[df['book'].astype(int).isin(included_books)]
            df = df.reset_index(drop=True)
            print(f"incl_books, {included_books}, dflen {len(df)}, df {df}")


        wordmap: dict = {w: i for i, w in enumerate(df['lemma'])}
        lemmas: pd.Series = pd.Series(wordmap.keys(), index = list(range(len(wordmap))) ) 
        lemma_index = pd.Index(lemmas)

        word_count = len(df['lemma'])
        lemmalen = len(lemmas)
        
        comat = np.zeros((lemmalen,lemmalen), dtype=int)
        for idx, lemma in enumerate(df['lemma']):
            for j in range(max(0, idx - window_size), min(word_count, idx + window_size + 1)):
                comat[lemma_index.get_loc(lemma)][lemma_index.get_loc(df['lemma'][j])] += 1
            
        return pd.DataFrame(comat, index=lemmas, columns=lemmas)            
        
    def process_book_input(self, book: str) -> str:
        return "00"

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
                                words_to_exclude: list[str] | None = None
                                ):
        # Make sure the graph doesn't have stuff already in it
        # print(f"calling with num_steps = {num_steps}, word = {search_word}")
#         # Initialize an empty list passed to recursive calls to avoid readding words
        if words_to_exclude is None:
            words_to_exclude = [search_word]
        if num_steps > 0:
            most_similar: pd.Series = self._most_similar(algo, search_word, df, words_per_level)
            print(f"Most similar to {search_word}: \n{most_similar}")
            for rel_word, similarity in zip(most_similar.index, most_similar):
                print(f"found {rel_word}")
                self.dg.add_weighted_edges_from([(search_word, rel_word, similarity)])
                if rel_word not in words_to_exclude:
                    words_to_exclude.append(rel_word)
                    self._build_word_search_network(df,
                                                    algo,
                                                    rel_word,
                                                    num_steps-1,
                                                    words_per_level,
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
        df = None

        retrain = False
        if 'retrain' in kwargs:
            retrain = kwargs['retrain']

        word = _self.process_strongs_input(unparsed_word[0], int(unparsed_word[1:])) #type: ignore
        source = Source[unparsed_word[0]]

        if algo == Algorithm.CON:
            df = _self.generate_comat(source, included_books=books_to_include)            
        elif algo == Algorithm.W2V:
            df = _self._generate_w2v_similarity_matrix(source=source, retrain=retrain) 
        else:
            raise NotImplementedError
        assert(df is not None)
        print(df)
        _self._build_word_search_network(df,
                                        algo,
                                        word,
                                        num_steps,
                                        words_per_level,
                                        words_to_exclude=None
                                        )
    
    def get_network(self) -> nx.DiGraph:
        return self.dg
