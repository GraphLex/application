
import networkx as nx
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim import utils

class VocabNet():
    def __init__(self):
        # Create a directed graph
        self.dg = nx.DiGraph()
        # Set a bool to track if the graph has had edges added to it
        self.populated = False

    #Use Keyman or lexilogos.
    #Put in hebrew words
    def word_search(self,
                    vecs: KeyedVectors,
                    word: int | str,
                    num_steps=3,
                    already_checked=None,
                    words_per_level=2) -> bool:
        # Make sure the graph doesn't have stuff already in it
        

        # Initialize an empty list passed to recursive calls to avoid readding words
        if already_checked is None:
            already_checked = []
            assert self.populated == False
            self.populated = True

        # print(f'called with num_steps = {num_steps}')

        # Check type of word param and find the correct word
        parsed_word: str = ""
        if type(word) == int:
            print(len(vecs.index_to_key))
            parsed_word = vecs.index_to_key[word]
        if type(word) == str:
            try:
                if vecs.key_to_index[word]:
                    parsed_word = word
            except Exception as e:
                print(type(e), e)
                return False
            
    
        most_similar = vecs.most_similar(parsed_word, topn=words_per_level)
        for v, similarity in most_similar:
            self.dg.add_weighted_edges_from([(parsed_word, v, similarity)])

            if num_steps is None:
                num_steps = 0

            if num_steps > 0:
                already_checked.append(vecs.key_to_index[v])
                self.word_search(vecs, vecs.key_to_index[v], num_steps-1, already_checked)
        return True


    def add_edges(self, vecs: KeyedVectors, max_count=200):
        assert not self.populated
        self.populated = True

        counter = 0
        for v1, num in vecs.key_to_index.items():
            for i in range(num):
                inner_counter = 0
                v2 = vecs.index_to_key[i]
                print(v1, v2)
                similarity = float(vecs.similarity(v1, v2))
                if -0.5 <= similarity >= 0.5:
                    self.dg.add_weighted_edges_from([(v1, v2, similarity)])
                    inner_counter += 1
                    if inner_counter >= max_count:
                        break
            counter += 1
            if counter >= max_count:
                break


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
    vn.add_edges(word_vectors)
