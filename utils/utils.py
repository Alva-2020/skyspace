#! /usr/bin/python
# -*- encoding: utf-8 -*-
import pdb
import math
import torch
import torch.nn.functional as F
import bisect 
'''
mount dropout '0.0@0.0,0.0@0.20,0.1@0.50,0.0@1.0'

plain dropout  contant float value
'''
def dropout_schedule(dropout_string, cur_epoch, tot_epoch):
    '''
    #mount dropout '0.0@0.0,0.0@0.20,0.1@0.50,0.0@1.0'
    schedule = []
    propors = []
    if "@" in name:
       for item in name.split(","):
           its = item.split("@")
           propors.append(float(its[1]))
           schedule.append((its[0],its[1]))
       proporition = current_epoch / epochs
       #
       index_next = bisect.bisect(propors,proporition)
       print(index_next)
       dropout = schedule[index_next-1][0]
       #print(dropout)
      
    elif sum([n.isdigit() for n in name.strip().split('.')]) == 2:

         dropout = float(name)
    return float(dropout)
    '''
    schedule = list()
    for string_list in dropout_string.split(','):
        point = string_list.split('@')
        schedule.append((float(point[0]), float(point[1])))

    rate = (cur_epoch+1)/tot_epoch

    for i in range(1, len(schedule)+1):
        if rate > schedule[i-1][1] and rate <= schedule[i][1]:
            dropout = schedule[i-1][0] + (schedule[i][0]-schedule[i-1][0])*((rate-schedule[i-1][1])/(schedule[i][1]-schedule[i-1][1]))
            return dropout
       
       
def learning_rate_schedule(current_epoch,epochs,initial_lr,name="inverse_curve",final_lr=1e-5,step=5):

    if name == "inverse_curve":

        return initial_lr * math.exp(current_epoch * math.log(final_lr/initial_lr) / epochs)
       
    elif name == "ladder":

        return max(initial_lr * (0.1** (current_epoch/step)),final_lr)
  
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)

    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    res = []

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
'''
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
'''

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert len(input.size()) == 2, 'The number of dimensions of input tensor must be 2!'
        # reflect padding to match lengths of in/out
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
if __name__ == '__main__':
    name="0@0.0,0@0.2,0.1@0.5,0@1.0"
    epochs = 30
    # learning_rate_schedule(t,100,0.1,name="inverse_curve",final_lr=1e-5,step=5)
    x = np.arange(0,30,1)
    y = []

    for t in range(epochs):
        lr = learning_rate_schedule(t,30,0.1,name="inverse_curve",final_lr=1e-5,step=5)
        #y.append(dropout_schedule(name,t,epochs))
        y.append(lr)
    plt.plot(x,y,label="dropout_schedule")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(0,0.1)
    plt.legend()
    plt.savefig("a.png")
        
     
