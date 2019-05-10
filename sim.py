import numpy as np
from scipy.spatial.distance import pdist, squareform
import sys




def main(vec_path,k):
    vecs = np.load(vec_path).item()
    idx2word = {idx:word for idx,word in enumerate(vecs.keys())}
    word2idx = {v:k for k,v in idx2word.items()}
    vecs_array = np.vstack([vecs[w].flatten() for w in vecs.keys()])

    idx2word = {}
    
    vecs_array = []
    for w,vec in vecs.items():
        for s_id, sense in enumerate(vec):
            idx2word[len(vecs_array)] = w+'_{}'.format(s_id)
            vecs_array.append(sense)

    vecs_array = np.array(vecs_array)
    word2idx = {v:k for k,v in idx2word.items()}

    dists = squareform(pdist(vecs_array, 'cosine'))
    # dists = squareform(pdist(vecs_array, 'euclidean'))
    topk = np.argsort(dists, axis=1)[:,:k]

    word = 'kings'

    word1 = word + '_0'
    print(word1)
    idx = word2idx[word1]
    idxs = topk[idx]
    words = [(idx2word[i], round(dists[idx,i],2)) for i in idxs]
    print(words[1:])


    word2 = word + '_1'
    print(word2)
    idx = word2idx[word2]
    idxs = topk[idx]
    words = [(idx2word[i], round(dists[idx,i],2)) for i in idxs]
    print(words[1:])


if __name__ == '__main__':
    if len(sys.argv) == 1:
        vec_path = 'w2vs_vecs.npy'
        k = 5
    elif len(sys.argv) ==2:
        vec_path = sys.argv[1]
        k = 5
    else:
        assert len(sys.argv) == 3
        vec_path = sys.argv[1]
        k = int(sys.argv[2])

    main(vec_path, k)