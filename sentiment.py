import torch
import torch.nn as nn
import torchtext
import numpy as np
from data import w2v_embedder, w2vs_embedder, SLS_Dataset
from torch.utils.data import DataLoader
from copy import deepcopy



def mean_len(sents, mask):
    # print(mask)
    return np.sum([len(s)*m for s,m in zip(sents, mask)])/np.sum(mask.detach().numpy())



def get_acc(data_loader, embedder, model, verbose=False):
    model = model.eval()
    correct = 0.0
    for data in data_loader:
        sents, labels = data
        embs, true_labels, true_sents = [], [], []
        for s,l in zip(sents, labels):
            emb = embedder(s)
            if emb is not None:
                embs.append(emb)
                true_labels.append(l)
                true_sents.append(s)
        labels = torch.FloatTensor(true_labels)
        embs = torch.stack(embs, dim=0)
        preds = model(embs)
        preds = torch.sigmoid(preds)
        preds = (preds >= 0.5).type(torch.float).squeeze()
        correct += torch.mean((preds == labels).type(torch.float))

        if verbose:
            # for sent, pred, label in zip(true_sents, preds, labels):
                # if label != pred:
                #     print(sent, 'PRED:', pred, 'LABEL:', label)
            wrong_mask = (preds != labels).type(torch.float)
            false_positives = torch.sum(wrong_mask*preds)/len(preds)
            false_negatives = torch.sum(wrong_mask*(1-preds))/len(preds)
            print('False positive rate:', false_positives)
            print('False negative rate:', false_negatives)

            correct_len = mean_len(true_sents, 1.0-wrong_mask)
            wrong_len = mean_len(true_sents, wrong_mask)

    model = model.train()
    return correct/len(data_loader)



def main():

    torch.manual_seed(0)
    np.random.seed(0)


    train_dataset = SLS_Dataset('data/sentiment_labelled_sentences/combined.txt',
                                 mode='train', splits=[0.8,0.1,0.1])

    val_dataset = SLS_Dataset('data/sentiment_labelled_sentences/combined.txt',
                                 mode='val', splits=[0.8,0.1,0.1])

    test_dataset = SLS_Dataset('data/sentiment_labelled_sentences/combined.txt',
                                 mode='test', splits=[0.8,0.1,0.1])

    train_loader = DataLoader(train_dataset, batch_size=100)
    val_loader = DataLoader(val_dataset, batch_size=300)
    test_loader = DataLoader(test_dataset, batch_size=300)



    w2v_emb = w2v_embedder('w2v_vecs.npy')


    names = ['w2v', 'w2vs_random', 'w2vs_greedy', 'w2vs_mean']
    embedders = [w2v_emb]
    for decoder in ['random', 'greedy', 'mean']:
        embedders.append(w2vs_embedder('w2vs_vecs.npy', decoder=decoder))


    for name, embedder in zip(names, embedders):
        print(name)




        in_dim = embedder.d
        # model = torch.nn.Sequential(nn.Linear(in_dim, 1))

        h_dim = 200
        model = torch.nn.Sequential(nn.Linear(in_dim, h_dim), nn.Linear(h_dim, 1))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        num_epochs = 75

        best_val_acc = -np.inf
        best_model = None

        val_accs = []

        val_acc = get_acc(val_loader, embedder, model)
        val_accs.append(float(val_acc))

        for i in range(num_epochs):
            epoch_loss = 0.0
            for data in train_loader:
                sents, labels = data
                embs, true_labels = [], []
                for s,l in zip(sents, labels):
                    emb = embedder(s)
                    if emb is not None:
                        embs.append(emb)
                        true_labels.append(l)
                labels = torch.FloatTensor(true_labels)
                embs = torch.stack(embs, dim=0)
                optimizer.zero_grad()
                preds = model(embs).squeeze()
                loss = criterion(preds, labels)
                epoch_loss += float(loss)
                loss.backward()
                optimizer.step()
            # print('epoch_loss:', epoch_loss)
            val_acc = get_acc(val_loader, embedder, model)
            val_accs.append(float(val_acc))
            # print('val acc:', val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = get_acc(test_loader, embedder, model)
                best_model = deepcopy(model)
        print('best validation accuracy:', best_val_acc)
        print('corresponding test accuracy:', test_acc)


        acc = get_acc(test_loader, embedder, best_model, verbose=True)
        np.save(name+'.npy',np.array(val_accs))






if __name__ == '__main__':
    main()

