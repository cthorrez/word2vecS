import torch
from data import txt_dataset
from models import Word2VecS


def test_dataset():
    data12 = txt_dataset('data/news.2012.en.shuffled.txt', num_lines=2000000)


    w2v = Word2VecS(d=100,p=2,dataset=data12,min_count=500,k=4,c=4,lr=1e-7, fname='w2v_vecs.npy')
    w2v.train(3)

    w2vs = Word2VecS(d=200,p=1,dataset=data12,min_count=500,k=4,c=4,lr=1e-7, fname='w2vs_vecs.npy')
    w2vs.train(3)

if __name__ == '__main__':
    test_dataset()
