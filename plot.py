import numpy as np
import matplotlib.pyplot as plt



def main():

    greedy = np.load('w2vs_greedy.npy')
    w2v = np.load('w2v.npy')
    random = np.load('w2vs_random.npy')
    mean = np.load('w2vs_mean.npy')


    x = np.arange(len(greedy))

    plt.plot(x, w2v, label='word2vec', color='black')
    plt.plot(x, greedy, label='word2vecS greedy', color='green')
    plt.plot(x, random, label='word2vecS random', color='orange')
    plt.plot(x, mean, label='word2vecS mean', color='blue')


    plt.xlabel('epoch')
    plt.ylabel('validation accuracy')
    plt.legend()
    plt.show()






if __name__ == '__main__':
    main()