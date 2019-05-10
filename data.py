import torch
import numpy as np
from torch.utils.data import Dataset
import re
import string
import itertools
from scipy.special import expit as sigmoid
from scipy.spatial.distance import cosine as cos_dist
from scipy.spatial.distance import cdist, squareform
import os.path as osp
from sklearn.model_selection import train_test_split



def process_sentence(s):
    s = s.lower()
    s = re.sub(r'[^a-z ]+', '',s)
    return s


class txt_dataset(Dataset):
    def __init__(self, fpath,num_lines=np.inf):
        super(txt_dataset, self).__init__()
        with open(fpath) as f:
            self.data = [process_sentence(s) for idx,s in enumerate(f.readlines()) if idx < num_lines]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class w2v_embedder:
    def __init__(self, vec_path):
        self.vecs = np.load(vec_path).item()

        self.d = list(self.vecs.values())[0].shape[1]
        self.idx2word = {idx:word for idx,word in enumerate(self.vecs.keys())}
        self.word2idx = {v:k for k,v in self.idx2word.items()}

    def __call__(self, sent):
        sent = sent.split()
        embs = []
        for word in sent:
            if word in self.word2idx:
                embs.append(self.vecs[word].squeeze())
        if len(embs) >= 1:
            mean = np.mean(embs, axis=0)
            return torch.tensor(mean)
        else:
            return None

def score_vecs(vecs):
    vec_list = [vecs[0].T]
    for v in vecs[1:-1]:
        vec_list.extend([v,v.T])
    vec_list.append(vecs[-1])
    score = 1.0
    for i in range(len(vec_list) - 1):
        score = score * sigmoid(np.dot(vec_list[i], vec_list[i+1]))
    return score


def greedy_decoder(words, vecs):
    out_vecs = [0]*len(words)
    if len(words) > 1:
        initial_scores = np.dot(vecs[words[0]], vecs[words[1]].T)
        idxs = np.unravel_index(np.argmax(initial_scores), initial_scores.shape)

        first_idx = idxs[0]
        second_idx = idxs[1]

        better_idx = first_idx
        out_vecs.append(vecs[words[0]][better_idx,:])

        for i in range(len(words) -1):
            scores = np.dot(vecs[words[i+1]], vecs[words[i]][better_idx,:])
            better_idx = np.argmax(scores)
            out_vecs.append(vecs[words[i+1]][better_idx,:])
        mean = np.mean(out_vecs, axis=0)
    else:
        np.random.seed(0)
        rand_idx = np.random.randint(low=0, high=2)
        mean = vecs[words[0]][rand_idx]
    return torch.tensor(mean)



def random_decoder(words, vecs):
    np.random.seed(0)
    out_vecs = [0]*len(words)
    for i in range(len(words)):
        idx = np.random.randint(low=0, high=2)
        out_vecs.append(vecs[words[i]][idx,:])
    mean = np.mean(out_vecs, axis=0)
    return torch.tensor(mean)


def mean_decoder(words, vecs):
    out_vecs = [0]*len(words)
    for i in range(len(words)):
        out_vecs.append(np.mean(vecs[words[i]], axis=0))
    mean = np.mean(out_vecs, axis=0)
    return torch.tensor(mean)


class w2vs_embedder:
    def __init__(self, vec_path, decoder='greedy'):
        self.vecs = np.load(vec_path).item()
        self.d = list(self.vecs.values())[0].shape[1]
        self.idx2word = {idx:word for idx,word in enumerate(self.vecs.keys())}
        self.word2idx = {v:k for k,v in self.idx2word.items()}

        if decoder == 'greedy':
            self.decoder = greedy_decoder
        elif decoder == 'random':
            self.decoder = random_decoder
        elif decoder == 'mean':
            self.decoder = mean_decoder
        else:
            print('use a valid decoder!')
            exit(1)

    def __call__(self, sent):
        sent = sent.split()
        words = []
        for word in sent:
            if word in self.vecs:
                words.append(word)
        if len(words) >= 1:
            mean = self.decoder(words, self.vecs)
            return mean
        else:
            return None


class SLS_Dataset(Dataset):
    def __init__(self, sent_path, mode, splits=[0.8,0.1,0.1], seed=0):
        super(SLS_Dataset, self).__init__()
        sentences = []
        labels = []
        with open(osp.join(sent_path)) as f:
            for line in f.readlines():
                line = line.strip()
                sent = line[:-1].strip()
                label = float(line[-1])
                sentences.append(process_sentence(sent)) 
                labels.append(label)

        x_train, x_test, y_train, y_test = train_test_split(sentences, labels, 
                                           test_size=1-splits[0], random_state=seed)
        test_size = round(splits[2] / (1.0-splits[0]), 1)
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_size, random_state=seed)
        if mode == 'train':
            self.sentences, self.labels = x_train, y_train
        elif mode == 'val':
            self.sentences, self.labels = x_val, y_val
        elif mode == 'test':
            self.sentences, self.labels = x_test, y_test

        self.size = len(self.sentences)

    def __getitem__(self,idx):
        return self.sentences[idx], self.labels[idx]


    def __len__(self):
        return self.size



def cos_sim01(x,y, use_max=True):
    dist = cdist(x,y, metric='cosine')
    if use_max:
        cos_sim = 1 -np.min(dist)
    else:
        cos_sim = 1 -np.max(dist)
    return (cos_sim + 1)/2

class sim:
    def __init__(self, vec_path, use_max=True):
        self.vecs = np.load(vec_path).item()
        self.use_max = use_max

    def __contains__(self, x):
        return x in self.vecs

    def __call__(self, w1, w2):
        return cos_sim01(self.vecs[w1], self.vecs[w2], self.use_max)




class sense_probe_dataset(Dataset):
    def __init__(self, text_path, vec_path, embedder, num_lines=np.inf):
        super(sense_probe_dataset, self).__init__()
        f = open(text_path)
        self.data = []
        for sent in f.readlines():
            sent = process_sentence(sent)
            sent = sent.split()
            for word in sent:
                if word in embedder:
                    pass

        self.vecs = np.load(vec_path)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

            
            

if __name__ == '__main__':
    data12 = txt_dataset('data/news.2012.en.shuffled.txt')