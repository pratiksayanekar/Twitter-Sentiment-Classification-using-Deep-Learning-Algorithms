import numpy as np
from gensim.models import Word2Vec


class Word2VecModel:
    def __init__(self):
        print("Word2Vec Model Initialization...")

    def create_model(self, X_train):
        # Creating Word2Vec training dataset.
        Word2vec_train_data = list(map(lambda x: x.split(), X_train))

        word2vec_model = Word2Vec(Word2vec_train_data,
                                  size=self.Embedding_dimensions,
                                  workers=8,
                                  min_count=5)
        # please refer https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4

        print("Vocabulary Length:", len(word2vec_model.wv.vocab))
        return word2vec_model

    def create_embedding_matrix(self, vocab_length, tokenizer, word2vec_model, max_len):
        embedding_matrix = np.zeros((vocab_length, max_len))

        for word, token in tokenizer.word_index.items():
            if word2vec_model.wv.__contains__(word):
                embedding_matrix[token] = word2vec_model.wv.__getitem__(word)

        print("Embedding Matrix Shape:", embedding_matrix.shape)
        return embedding_matrix