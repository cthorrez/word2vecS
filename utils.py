import numpy as np
from itertools import compress

class NegativeSampler:
    def __init__(self, vocab, seed=0):
        np.random.seed(seed)
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.words_ordered = ['']*len(vocab)
        probs = np.zeros(self.vocab_size)
        for word, (idx ,count) in vocab.items():
            probs[idx] = np.power(count,0.75)
            self.words_ordered[idx] = word
        total = np.sum(probs)
        self.probs = probs/total
        self.probs[-1] = 1.0 - np.sum(self.probs[:-1])
        assert np.sum(self.probs) == 1.0

    def sample(self, n=1, words=True):
        if words:
            return np.random.choice(self.words_ordered, size=n, replace=True, p=self.probs)
        else:
            return np.random.choice(self.vocab_size, size=n, replace=True, p=self.probs)


class SubSampler:
    def __init__(self, vocab, t=5e-5, seed=0):
        np.random.seed(seed)
        tot_words = np.sum([v[1] for v in vocab.values()])
        self.vocab = vocab
        self.probs = np.zeros(len(vocab))
        for w, (idx,count) in vocab.items():
            self.probs[idx] = count
        self.probs = 1.0 - np.sqrt(t*(tot_words/self.probs))
        print('min, mean, max drop prob:',np.min(self.probs), np.mean(self.probs), np.max(self.probs))


    # takes in a sentence and removes some of the common words
    # rules described in the 2.3 of Mikolov et al 2013
    def subsample_sent(self, sent):
        idx_sent = np.array([self.vocab[w][0] for w in sent])
        sent_probs = self.probs[idx_sent]
        keeps = np.random.rand(*sent_probs.shape) > sent_probs
        return list(compress(sent,keeps))
        

# given a sentence and a context length get pairs within context length
def get_pairs(sent, c):
    length = len(sent)
    pairs = []
    for i in range(length):
        word = sent[i]
        l = max(0, i - c)
        r = min(length, i+c+1)
        for j in range(l,r):
            if i != j:
                pairs.append((sent[i], sent[j]))
    return pairs

