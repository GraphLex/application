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

    def _initialize_df(self, lemmas: pd.Series) -> pd.DataFrame:
        # Little bit of a conceptual help from ChatGPT on dataframe initialization
        # TODO: Is there a more efficient way to do this?
        df = pd.DataFrame(index=lemmas, columns=lemmas, dtype=int)
        for col in df.columns:
            df[col] = 0
        return df

    def _most_similar(self, algo: Algorithm, word: str, df: pd.DataFrame, topn: int) -> pd.Series:
        match algo:
            case Algorithm.CON:
                # df is the con matrix
                return df[word].nlargest(topn)               
            case Algorithm.W2V:
                return pd.Series(df[word].nlargest(topn))


    def generate_full_comat(self, code: Literal['H', 'G'], window_size = 3) -> pd.DataFrame:
        lemmas = None
        dataset = None
        if code == 'H':
            # Calculate unique lemmas from the the full dataset
            lemmas = pd.Series(list(set(self.hb[2])))
            dataset = self.hb
        elif code == 'G':
            lemmas = pd.Series(list(set(self.gnt[3])))
            dataset = self.gnt
        else:
            # Janky, but it works
            raise ValueError("Incorrect testament code")
            
        df = self._initialize_df(lemmas)
        start_index = 10
        arr = np.ndarray((len(lemmas), len(lemmas)), dtype=int)
        rows_to_add = []
        cols_to_add = []
        for i in range(len(dataset) - 2 * start_index): # originally had lemma w/ an enumerate here
            lower_bound = i - window_size + start_index + 1
            upper_bound = i + window_size + start_index + 1
            current_lemma = dataset[i+start_index]
            window_slice = dataset[lower_bound:upper_bound]

            for co_oc_lemma in window_slice:
                # df.iat[current_word, co_oc_lemma] += 1 # type: ignore
                
                rows_to_add.append(lemmas.get_loc(current_lemma))
                cols_to_add.append(lemmas.get_loc(co_oc_lemma))

        np.add.at(arr, (rows_to_add, cols_to_add), 1)
        df = pd.DataFrame()
        print(df.head)
        return df
        # self.dg = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
        
    def generate_single_comat(self, target_word: str, code: Literal['H', 'G'], window_size = 3) -> pd.DataFrame:
        lemmas = None
        dataset = None
        if code == 'H':
            # Calculate unique lemmas from the the full dataset
            lemmas = pd.Series(list(set(self.hb['lemma'])))
            dataset = self.hb
        elif code == 'G':
            lemmas = pd.Series(list(set(self.gnt['lemma'])))
            dataset = self.gnt
        else:
            # Janky, but it works
            raise ValueError("Incorrect testament code")
            
        df = self._initialize_df(lemmas)
        start_index = 10

        for i in range(len(dataset[start_index:-start_index])):
            current_word = dataset.loc[i+start_index]['lemma']
            print(type(current_word))

            if current_word == target_word:
                
                lower_bound = i - window_size + start_index + 1
                upper_bound = i + window_size + start_index + 1
                
                window_slice = dataset.loc[lower_bound:upper_bound]['lemma']
                print(window_slice)
            

                for co_oc_lemma in window_slice:
                    print(current_word, co_oc_lemma)
                    df.at[current_word, co_oc_lemma] += 1 # type: ignore
            
        return df
            
        
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
        
    def _generate_w2v_similarity_matrix(self, source: Source, retrain: bool = False) -> pd.DataFrame:
        if retrain:
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
        print(f"calling with num_steps = {num_steps}, word = {search_word}")
#         # Initialize an empty list passed to recursive calls to avoid readding words
        if words_to_exclude is None:
            words_to_exclude = [search_word]
        if num_steps > 0:
            most_similar: pd.Series = self._most_similar(algo, search_word, df, words_per_level)
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

        word = _self.process_strongs_input(unparsed_word[0], int(unparsed_word[1:]))

        if algo == Algorithm.CON:
            df = _self.generate_single_comat(word, 'H')
            
    
        elif algo == Algorithm.W2V:
            print("Running W2V...")
            print(f"Unparsed word = {unparsed_word}, {unparsed_word[0]}")
            df = _self._generate_w2v_similarity_matrix(source=Source[unparsed_word[0]], retrain=retrain) 
            print("building network")
        else:
            raise NotImplementedError
        assert(df is not None)
        _self._build_word_search_network(df,
                                        algo,
                                        word,
                                        num_steps,
                                        words_per_level,
                                        words_to_exclude=None
                                        )
    
    def get_network(self) -> nx.DiGraph:
        return self.dg



# class VocabNet():
#     def __init__(self):
#         # Create a directed graph
#         self.dg = nx.DiGraph()
#         # Set a bool to track if the graph has had edges added to it
#         self.populated = False

#     def word_search(self,
#                     vecs: KeyedVectors,
#                     word: int | str,
#                     num_steps=3,
#                     words_to_exclude: list[int] | None = None,
#                     words_per_level=2) -> bool:
#         '''Build a graph by recursively adding the node's nearest neighbors.
#         :param vecs: A KeyedVectors object containing the word embedding vectors.
#         :param word: An int indicating the index of the word, or a string containing the word for the node to search.
#         :param num_steps: the number of recursive steps to take.
#         :param words_to_exclude: a list of integer indices specifying words to exclude -
#             do not add a value to this parameter from outside the function; for internal purposes only.
#         :param words_per_level: the number of nearest neighbors to add to the graph at each level.
        
#         '''

#         # Make sure the graph doesn't have stuff already in it
        

#         # Initialize an empty list passed to recursive calls to avoid readding words
#         if words_to_exclude is None:
#             words_to_exclude = []
#             assert self.populated == False
#             self.populated = True

#         # print(f'called with num_steps = {num_steps}')

#         # Check type of word param and find the correct word
#         parsed_word: str = ""
#         if type(word) == int:
#             print(len(vecs.index_to_key))
#             parsed_word = vecs.index_to_key[word]
#         if type(word) == str:
#             try:
#                 if vecs.key_to_index[word]:
#                     parsed_word = word
#             except Exception as e:
#                 print(type(e), e)
#                 return False
            
    
#         most_similar = vecs.most_similar(parsed_word, topn=words_per_level)
#         for v, similarity in most_similar:
#             self.dg.add_weighted_edges_from([(parsed_word, v, similarity)])
#             if num_steps > 0:
#                 words_to_exclude.append(vecs.key_to_index[v])
#                 self.word_search(vecs, vecs.key_to_index[v], num_steps-1, words_to_exclude)
#         return True


#     def add_edges(self, vecs: KeyedVectors, max_count=200):
#         assert not self.populated
#         self.populated = True

#         counter = 0
#         for v1, num in vecs.key_to_index.items():
#             for i in range(num):
#                 inner_counter = 0
#                 v2 = vecs.index_to_key[i]
#                 print(v1, v2)
#                 similarity = float(vecs.similarity(v1, v2))
#                 if -0.5 <= similarity >= 0.5:
#                     self.dg.add_weighted_edges_from([(v1, v2, similarity)])
#                     inner_counter += 1
#                     if inner_counter >= max_count:
#                         break
#             counter += 1
#             if counter >= max_count:
#                 break



if __name__ == "__main__":
    
    class KJV():

        def _setup_df(self):
            verse_lists = [verse.split(" ")[2:-4] for verse in list(self.df['response'])]
            self.df['response'] = [" ".join(a) for a in verse_lists]
             
        def __init__(self, df):
            self.df = df
            self._setup_df()

        def __iter__(self):
            for response in self.df['response']:
                yield utils.simple_preprocess(response)

    vn = VocabNet()
    if (input("Retrain model? y/n ") == "y"):
        corpus = KJV(pd.read_json("hf://datasets/oliverbob/openbible/bible.json"))
        model = Word2Vec(sentences=corpus)
        word_vectors = model.wv
        del model
        word_vectors.save("stored_nt_vectors.kv")
    else:
        word_vectors = KeyedVectors.load("stored_nt_vectors.kv")
    # vn.add_edges(word_vectors)
