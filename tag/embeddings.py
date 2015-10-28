import numpy as np

from gensim.models import Word2Vec

class SentenceIterator:
    def __init__(self, tokens,
                 sentence_len=100):
        self.sentence_len = sentence_len
        self.tokens = tokens
        self.idxs = []
        start_idx, end_idx = 0, self.sentence_len
        while end_idx < len(self.tokens):
            self.idxs.append((start_idx, end_idx))
            start_idx += self.sentence_len
            end_idx += self.sentence_len

    def __iter__(self):
        for start_idx, end_idx in self.idxs:
            yield self.tokens[start_idx : end_idx]


class EmbeddingsPretrainer:
    def __init__(self, tokens,
                 sentence_len=100,
                 window=10,
                 min_count=1,
                 size=300,
                 workers=10,
                 negative=5):
        self.size = size
        sentence_iterator = SentenceIterator(sentence_len=sentence_len,
                                             tokens=tokens)
        self.w2v_model = Word2Vec(sentence_iterator,
                             window=window,
                             min_count=min_count,
                             size=self.size,
                             workers=workers,
                             negative=negative)

    def get_weights(self, vocab):
        unk = np.zeros(self.size)
        weights = []
        for w in vocab:
            try:
                weights.append(self.w2v_model[w])
            except KeyError:
                weights.append(unk)
        return [np.asarray(weights)]