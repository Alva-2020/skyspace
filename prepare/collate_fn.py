import random
import pdb
import torch
import numpy as np

def random_collate(data):
    
    min_len = min(data, key=lambda x:x[0].shape[0])[0].shape[0]
    
    data_numpy = np.zeros((len(data), min_len, data[0][0].shape[1]), dtype='float32')
    # waste time operation with iteration whole batchsize
    for i in range(len(data)):

        free_length = data[i][0].shape[0] - min_len

        offset = random.randint(0, free_length)

        data_numpy[i] = data[i][0][offset:min_len+offset]

    data_tensor = torch.from_numpy(data_numpy)
    
    label_tensor = torch.LongTensor([data[i][1] for i in range(len(data))])
    
    return data_tensor, label_tensor


if __name__ =="__main__":
    data = torch.Tensor(3,4,5)
    pdb.set_trace()
    min_len = min(data, key=lambda x : x[0].shape[0])
