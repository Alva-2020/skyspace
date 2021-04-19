import pdb
import torch
import torch.nn as nn

#statistic pooling
class statistic_pooling(nn.Module):
    def __init__(self):
        super(statistic_pooling,self).__init__()

    def forward(self,feature):
        if len(feature.size()) > 3:
            feature = feature.view(feature.size()[0],feature.size()[1],-1)
        assert len(feature.size()) == 3
        mean = feature.mean(dim=2)
        std = feature.std(dim=2)
        stats = torch.cat((mean, std),dim=1)
        return stats

class statistic_att_pooling(nn.Module):
    def __init__(self,input_dim,middle_dim,pooling="SAP"):
       
        super(statistic_att_pooing,self).__init__()
        self.pooling = pooling
        self.attention = nn.Sequential( 
            nn.Conv1d(input_dim, middle_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(middle_dim),
            nn.Conv1d(middle_dim, input_dim, kernel_size=1),
            nn.Softmax(dim=2),
            )
        if self.pooling == "SAP":
           out_dim = input_dim
        elif self.pooling =="ASP":
           out_dim = input_dim * 2


    def forward(self,feature):

        w = self.attention(feature)

        if self.pooling == "SAP":

           x = torch.sum(feature * w, dim=2)

        elif self.pooling == "ASP":

           mu = torch.sum(feature * w, dim=2)

           sg = torch.sqrt( ( torch.sum((feature**2) * w, dim=2) - mu**2 ).clamp(min=1e-5) )
 
           x = torch.cat((mu,sg),1)
        return x 
         

if __name__ == '__main__':
    feature = torch.Tensor(8,512,3)
    statistic_pooling(feature).size()
