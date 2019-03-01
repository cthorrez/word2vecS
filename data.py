import torch
import numpy as np
from torch.utils.data import Dataset
import re
import string



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
            
            

if __name__ == '__main__':
    data12 = txt_dataset('data/news.2012.en.shuffled.txt')