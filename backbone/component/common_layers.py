import pdb
import torch
import torch.nn as nn

class conv1D(nn.Module):

   def __init__(self, input_dim, output_dim, kernel_size, dilation=1, stride=1):
       super(conv1D,self).__init__()
       self.conv1d = nn.Conv1d(input_dim, output_dim, kernel_size, dilation=dilation, stride=stride)
       # default affine = True, there is an akin affine layer after normlization.
       self.bn = nn.BatchNorm1d(output_dim)
       self.relu = nn.ReLU()

   def forward(self, input_x):
       x = self.conv1d(input_x)
       x = self.bn(x)
       x = self.relu(x)
       return x 

class conv2D(nn.Module):

   def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, dilation=1):
      super(conv2D,self).__init__()
      self.conv2d = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
      self.bn = nn.BatchNorm2d(input_dim)
      self.relu = nn.ReLU()
     
   def forward(self, input_x):
      x = self.conv2d(input_x)
      x = self.bn(x)
      x = self.relu(x)
      return x
      
       

class full_connected(nn.Module):

   def __init__(self, input_dim, output_dim, p_dropout=0.0):
   
       super(full_connected, self).__init__()
       self.linear = nn.Linear(input_dim, output_dim)
       self.bn = nn.BatchNorm1d(output_dim)
       self.relu = nn.ReLU()
       self.dropout = nn.Dropout(p=p_dropout) 
     
   def forward(self, input_x):
       x = self.linear(input_x)
       x = self.bn(x)
       x = self.relu(x)
       x = self.dropout(x)
       return x 
      #In training stage, change dropout proportion dynamically each epoch  
   def set_dropout(self, dropout):
       for layer in self.modules():
           if isinstance(layer, torch.nn.Dropout):
              print(layer.p)
                 

if __name__ == '__main__':
   full_linear = full_connected(2,10)
   full_linear.set_dropout(0.1)
   quit()
   model = tdnn(8,10,2)
   print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()]))) 
