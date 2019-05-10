import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse


from data import sim




def main():
    w2v_sim = sim('w2v_vecs.npy')
    w2vs_sim = sim('w2vs_vecs.npy', use_max=False)

    w1s = []
    w2s = []
    gt_sims1  =[]

    with open('data/SCWS/ratings.txt') as f:
        for idx,line in enumerate(f.readlines()):
            sep = line.lower().split('\t')
            w1s.append(sep[1])
            w2s.append(sep[3])
            gt_sims1.append(sep[7])


    comps = 0
    w2v_sims = []
    w2vs_sims = []
    gt_sims  =[]


    for w1, w2, gt_sim in zip(w1s,w2s,gt_sims1):
        if w1 in w2v_sim and w2 in w2v_sim and w1 in w2vs_sim and w2 in w2vs_sim:
            comps+=1
            w2v_sims.append(w2v_sim(w1,w2))
            w2vs_sims.append(w2vs_sim(w1,w2))
            gt_sims.append(float(gt_sim)/10)




    print('word2vec mse:', mse(w2v_sims, gt_sims))
    print('word2vecS mse:', mse(w2vs_sims, gt_sims))

    print('word2vec mae:', mae(w2v_sims, gt_sims))
    print('word2vecS mae:', mae(w2vs_sims, gt_sims))

    print(comps, 'comparisons out of 2003')



if __name__ == '__main__':
    main()