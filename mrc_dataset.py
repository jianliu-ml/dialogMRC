import copy
import sys
import torch
import pickle
import random
import numpy as np

from tqdm import tqdm


class Dataset(object):
    def __init__(self, batch_size, dataset):
        super().__init__()

        self.batch_size = batch_size
        self.construct_index(dataset)

    def construct_index(self, dataset):
        self.dataset = dataset
        self.index_length = len(dataset)
        self.shuffle_list = list(range(0, self.index_length))

    def shuffle(self):
        random.shuffle(self.shuffle_list)

    def get_tqdm(self, device, shuffle=True):
        return tqdm(self.reader(device, shuffle), mininterval=2, total=self.index_length // self.batch_size, leave=False, file=sys.stdout, ncols=80)

    def reader(self, device, shuffle):
        cur_idx = 0
        while cur_idx < self.index_length:
            end_index = min(cur_idx + self.batch_size, self.index_length)
            batch = [self.dataset[self.shuffle_list[index]] for index in range(cur_idx, end_index)]
            cur_idx = end_index
            yield self.batchify(batch, device)
        if shuffle:
            self.shuffle()

    def batchify(self, batch, device):

        data_x, data_mask, data_seg = list(), list(), list()
        starts, ends = list(), list()

        for data in batch:
            data_x.append(data[0])
            data_mask.append(data[1])
            data_seg.append(data[2])
            starts.append(data[3])
            ends.append(data[4])

        f = torch.LongTensor
        
        data_x = f(data_x)
        data_mask = f(data_mask)
        data_seg = f(data_seg)
        data_start = f(starts)
        data_end = f(ends)

        return [data_x.to(device),  
                data_mask.to(device),
                data_seg.to(device),
                data_start.to(device),
                data_end.to(device)]


if __name__ == "__main__":
    
    with open('data/data.pickle', 'rb') as f:
        [train_examples, dev_examples, test_examples] = pickle.load(f)

    train_all = []
    for elem in train_examples:
        for e in elem[2]:
            if e:
                train_all.append(e)
    print(len(train_all))
    
    train_dataset = Dataset(20, train_all)
    for batch in train_dataset.reader('cpu', False):
        data_x, data_mask, data_seg, data_start, data_end = batch
        print(data_x)
        break
