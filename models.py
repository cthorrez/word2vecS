import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from utils import get_pairs, NegativeSampler, SubSampler




class Word2VecS():
    def __init__(self, d, p, dataset, min_count, k, c, lr, seed=0, fname='vecs.npy'):
        torch.manual_seed(seed)
        self.fname=fname
        self.d = d
        self.p = p
        self.dataset = dataset
        print(len(dataset),'sentences')
        print(np.sum([len(s.split()) for s in self.dataset]), 'words')
        self.min_count = min_count
        self.k = k
        self.c = c
        self.process_vocab() # dict word : (idx, count)
        self.vocab_size = len(self.vocab)
        self.neg_sampler = NegativeSampler(self.vocab)
        self.sub_sampler = SubSampler(self.vocab)

        self.vecs = torch.zeros((self.vocab_size,self.p,self.d), dtype=torch.float, requires_grad=True)
        nn.init.normal_(self.vecs,0,1e-2)

        self.sigmoid = nn.Sigmoid()
        # self.optimizer = torch.optim.SGD((self.vecs,), lr=lr)
        self.optimizer = torch.optim.Adam((self.vecs,), lr=lr)

    # sets self.vocab
    def process_vocab(self):
        print('processing vocab')
        words_counts = defaultdict(int)
        for sent in self.dataset:
            for w in sent.split():
                words_counts[w] += 1

        words_counts = {k:v for k,v in words_counts.items() if v>=self.min_count}        
        
        self.vocab = {}
        for idx, w in enumerate(words_counts.keys()):
            self.vocab[w] = (idx, words_counts[w])

        print('vocab_size:', len(self.vocab))
        

    def train(self, num_epochs):
        print('training')
        for epoch in range(num_epochs):
            loss = self.train_epoch()
            print('epoch: {} loss: {}'.format(epoch,loss))
        self.save()

    def train_epoch(self):
        epoch_loss = 0.
        for sent in self.dataset:
            sent = [w for w in sent.split() if w in self.vocab]
            if len(sent) <= 1:
                continue
            sent = self.sub_sampler.subsample_sent(sent)
            if len(sent) <= 1:
                continue
            pairs = get_pairs(sent, self.c)



            # sentence batched version
            loss = 0.
            pair_idxs = torch.tensor([[self.vocab[pair[0]][0],self.vocab[pair[1]][0]] for pair in pairs])

            inps = self.vecs[pair_idxs[:,0]]
            targs = self.vecs[pair_idxs[:,1]]
            t_scores = torch.bmm(inps,torch.transpose(targs,1,2))

            # find indices of the senses used for each word in the sentence
            m2, mi2 = torch.max(t_scores, dim=2)
            senses = torch.argmax(m2, dim=1)
            
            # This one also "negative samples" the scores produced by non maximal word sense pairings
            # _ , t_max_ind = torch.max(t_scores,dim=1)
            # t_scores *= -1
            # t_scores[np.arange(len(t_scores)),t_max_ind] *= -1
            # t_probs = self.sigmoid(t_scores)

            # This does not penalize other senses
            t_probs = self.sigmoid(torch.max(t_scores.view(len(pairs),-1),dim=1)[0])

            loss += -torch.mean(torch.log(t_probs))

            # can I also batch negative samples? :O
            neg_batch_idxs = self.neg_sampler.sample(self.k*len(pairs), words=False)
            ntargs = self.vecs[neg_batch_idxs]
            neg_senses = senses.repeat(self.k)
            neg_inps = inps.repeat(self.k,1,1)

            # only use the senses indicated by max score
            neg_inps = neg_inps[np.arange(len(neg_inps)),neg_senses,:]

            n_scores = torch.bmm(neg_inps.view(len(neg_inps),1,-1),torch.transpose(ntargs,1,2))

            #negative sample every sense pairing?
            # n_probs = self.sigmoid(-1.0*n_scores)

            # only negative sample the max?
            n_probs = self.sigmoid(-1.0*torch.max(n_scores.view(len(n_scores),-1),dim=1)[0])

            # only negative sample the sense that inp was actually used in. (the row which max occurs in)
            # n_probs = self.sigmoid(-1.0*n_scores)
            
            loss += -torch.sum(torch.log(n_probs))/len(n_probs)
            loss.backward()
            self.optimizer.step()
            epoch_loss += float(loss)


            
            # # non batched version
            # for pair in pairs:
            #     loss = 0.
            #     self.optimizer.zero_grad()

            #     inp = self.vecs[self.vocab[pair[0]][0]]
            #     targ = self.vecs[self.vocab[pair[1]][0]]

            #     t_scores = torch.mm(inp, targ.t())
            #     t_prob = self.sigmoid(torch.max(t_scores))
            #     loss += -torch.log(t_prob)

            #     # negative sampling
            #     neg_samples = self.neg_sampler.sample(self.k)
            #     for ns in neg_samples:
            #         ntarg = self.vecs[self.vocab[ns][0]]
            #         n_scores = torch.mm(inp, ntarg.t())
            #         n_prob = self.sigmoid(-1.0*torch.max(n_scores))
            #         loss += -torch.log(n_prob)

            #     loss.backward()
            #     self.optimizer.step()
            #     epoch_loss += float(loss)
        

        return epoch_loss


    def save(self):
        outdict = {}
        for word, (idx,_) in self.vocab.items():
            outdict[word] = self.vecs[idx].data.numpy()
        np.save(self.fname, outdict)







