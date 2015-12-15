from __future__ import print_function
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE


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
                 window=5,
                 min_count=1,
                 size=300,
                 workers=10,
                 negative=5,
                 nb_mfi=500):
        self.size = size
        self.mfi = [t for t,_ in Counter(tokens).most_common(nb_mfi)] # get most frequent items
        self.sentence_iterator = SentenceIterator(sentence_len=sentence_len,
                                             tokens=tokens)
        self.w2v_model = Word2Vec(self.sentence_iterator,
                             window=window,
                             min_count=min_count,
                             size=self.size,
                             workers=workers,
                             negative=negative)
        self.plot_mfi()
        self.most_similar()

    def plot_mfi(self, outputfile='embeddings.pdf', nb_clusters=8):
        # collect embeddings for mfi:
        X = np.asarray([self.w2v_model[w] for w in self.mfi \
                            if w in self.w2v_model], dtype='float64')
        # dimension reduction:
        tsne = TSNE(n_components=2)
        coor = tsne.fit_transform(X) # unsparsify

        plt.clf()
        sns.set_style('dark')
        sns.plt.rcParams['axes.linewidth'] = 0.4
        fig, ax1 = sns.plt.subplots()  

        labels = self.mfi
        # first plot slices:
        x1, x2 = coor[:,0], coor[:,1]
        ax1.scatter(x1, x2, 100, edgecolors='none', facecolors='none')
        # clustering on top (add some colouring):
        clustering = AgglomerativeClustering(linkage='ward', affinity='euclidean', n_clusters=nb_clusters)
        clustering.fit(coor)
        # add names:
        for x, y, name, cluster_label in zip(x1, x2, labels, clustering.labels_):
            ax1.text(x, y, name, ha='center', va="center",
                     color=plt.cm.spectral(cluster_label / 10.),
                     fontdict={'family': 'Arial', 'size': 8})
        # control aesthetics:
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        ax1.set_xticklabels([])
        ax1.set_xticks([])
        ax1.set_yticklabels([])
        ax1.set_yticks([])
        sns.plt.savefig(outputfile, bbox_inches=0)

    def most_similar(self, nb_neighbors=5,
                     words=['doet', 'goet', 'ende', 'mach', 'gode'],
                     outputfile='neighbours.txt'):
        with open(outputfile, 'w') as f:
            for w in words:
                try:
                    neighbors = ' - '.join([v for v,_ in self.w2v_model.most_similar(w)])
                    f.write(' '.join((w, '>', neighbors))+'\n')
                    f.write(':::::::::::::::::\n')
                except KeyError:
                    pass


    def get_weights(self, vocab):
        unk = np.zeros(self.size)
        weights = []
        for w in vocab:
            try:
                weights.append(self.w2v_model[w])
            except KeyError:
                weights.append(unk)
        return [np.asarray(weights)]