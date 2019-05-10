import numpy as np
from data import w2v_embedder, w2vs_embedder



def main():
    w2v_emb = w2v_embedder('300_1_35_3_2_5e-7_5.npy')
    w2vs_emb = w2vs_embedder('big_boi.npy')
    w2vs_emb = w2vs_embedder('most_recent.npy')

    sent = ['hello', 'i', 'want', 'to', 'go', 'to', 'the', 'river', 'bank', 'today']

    a = w2v_emb(sent)

    b = w2vs_emb(sent)
    


if __name__ == '__main__':
    main()


