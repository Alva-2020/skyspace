import torch
import torch.nn as nn

class CESoftmax(nn.Module):
    def __init__(self,embedding_size, numSpkrs):
        super(CESoftmax,self).__init__()
        self.fc_class = nn.Linear(embedding_size, numSpkrs)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,input_x, labels):
        x = self.fc_class(input_x)
        loss = self.criterion(x, labels)
        return x, loss

class AAMSoftmax(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes=10,
                 m=0.1,
                 s=15, 
                 easy_margin=False):
        super(AAMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_classes, in_feats), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        print('Initialised AMSoftmax m=%.3f s=%.3f'%(self.m,self.s))
    def set_hyparameters(self, m=0.2, s=30):
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label=None):
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss = self.ce(output, label)
        #prec1, _   = accuracy(output.detach().cpu(), label.detach().cpu(), topk=(1, 5))
        return loss, posterior
