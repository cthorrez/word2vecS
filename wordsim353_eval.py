import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse


from data import sim




def main():
    w2v_sim = sim('w2v_vecs.npy')
    w2vs_sim = sim('w2vs_vecs.npy')

    data = np.loadtxt('data/wordsim353/combined.csv', skiprows=1, delimiter=',' , dtype=np.str)


    comps = 0
    w2v_sims = []
    w2vs_sims = []
    gt_sims = []
    for w1, w2, gt_sim in data:
        if w1 in w2v_sim and w2 in w2v_sim and w1 in w2vs_sim and w2 in w2vs_sim:
            comps+=1
            w2v_sims.append(w2v_sim(w1,w2))
            w2vs_sims.append(w2vs_sim(w1,w2))
            gt_sims.append(float(gt_sim)/10)


    print('word2vec mse:', mse(w2v_sims, gt_sims))
    print('word2vecS mse:', mse(w2vs_sims, gt_sims))

    print('word2vec mae:', mae(w2v_sims, gt_sims))
    print('word2vecS mae:', mae(w2vs_sims, gt_sims))

    print(comps, 'comparisons out of 353')



if __name__ == '__main__':
    main()