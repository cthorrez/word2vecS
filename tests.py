import torch
from data import txt_dataset
from models import Word2VecS


def test_dataset():
    # data12 = txt_dataset('data/news.2012.en.shuffled.txt')
    data12 = txt_dataset('data/news.2012.en.shuffled.txt', num_lines=200000)
    w2v = Word2VecS(d=50,p=2,dataset=data12,min_count=50,k=3,c=2,lr=1e-6)

    w2v.train(3)

if __name__ == '__main__':
    test_dataset()
