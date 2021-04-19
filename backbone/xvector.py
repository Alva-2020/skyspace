import pdb
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.append("..")
from backbone.component.common_layers import conv1D, full_connected 

from loss.criterion import CESoftmax,AAMSoftmax
from pooling.pooling import statistic_pooling, statistic_att_pooling
 
'''
Embedding extarctor and loss function
'''
class xvecTDNN(nn.Module):

    def __init__(self, input_dim=80, 
                       output_channels=[512,512,512,512,1500],
                       kernels_size=[5,3,3,1,1],
                       dilations=[1,2,3,1,1],
                       pooling="STP", 
                       embedding_dim=512,
                       numSpkrs = 1211, 
                       p_dropout=0.0):

        super(xvecTDNN,self).__init__()

        self.pooling = pooling

        self.tdnn = nn.Sequential(
 
                    conv1D(input_dim, output_channels[0], kernels_size[0], dilation=dilations[0]),
                  
                    conv1D(output_channels[0], output_channels[1], kernels_size[1], dilation=dilations[1]),
                    conv1D(output_channels[1], output_channels[2], kernels_size[2], dilation=dilations[2]),
                    conv1D(output_channels[2], output_channels[3], kernels_size[3], dilation=dilations[3]),
                    conv1D(output_channels[3], output_channels[4], kernels_size[4], dilation=dilations[4])
                                  )
        if pooling == 'STP':
           self.pooling_layer = statistic_pooling()
           out_dim = output_channels[4] * 2   
         
      
        self.fc1 = full_connected(out_dim,embedding_dim, p_dropout=p_dropout)

        self.embedding = full_connected(embedding_dim, embedding_dim)

        self.ce_loss = CESoftmax(embedding_dim, numSpkrs)

    def forward(self, x, label=None, eps=1e-5):

        # x =[B,F,T]
        x = x.transpose(-1,-2).contiguous()

        x = self.tdnn(x)
 
        if self.training:
            shape = x.size()
            noise = torch.cuda.FloatTensor(shape)
            torch.randn(shape, out=noise)
            x += noise * eps

        if self.pooling == "STP":
           x = self.pooling_layer(x)
        x = self.fc1(x)

        x = self.embedding(x)
        
        embeddings = x
        
        postorior, loss = self.ce_loss(x,label)

        return embeddings,postorior, loss
    def set_dropout(self, dropout):
        #logger.info("change model's dropout to {}".format(dropout))
        for layer in self.modules():
            if isinstance(layer, torch.nn.Dropout):
                layer.p = dropout

    
