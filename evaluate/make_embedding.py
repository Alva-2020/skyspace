import argparse
import numpy as np
import torch.nn as nn
from MODELS.model_resnet import * 
from prepare_test_data import getTestData
import kaldi_io 
import pdb
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
import time

from backbone.xvector_v1 import xvecTDNN

class ExtractEmbedding(nn.Module):
    def __init__(self,model,extract_layers):
        super(ExtractEmbedding,self).__init__()
        self.model = model
        self.extract_layers = extract_layers

    def forward(self, x):
        tmp_x =x
        fw =open("model_struct","w")
        fw.write(str(self.model))
        for name , submodule in self.model._modules.items():

            for subname,submodel in submodule._modules.items():
                if subname is 'fc': break
                x = submodel(x)
            
        return x       
def load_model():
    #model = ResidualNet("ImageNet",18 ,7325, "BAM")
    model = xvecTDNN(1211,0.0)
    #path = "./checkpoints_100_clean/99.RESNET18_AUDIOvoxceleb_BAM_checkpoint.pth.tar"
    path="../checkpoints/4.xvector_voxceleb1_arcface_checkpoint.pth.tar"
    #model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    #path = "./checkpoints/RESNET50_IMAGENET_BAM_model_best.pth.tar"
    checkpoint = torch.load(path)
    start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    return model

def get_utt_xvector(utt2spk,featscp):
    #getTestDataset
    #getbatchsize()
    input_x,target_var = getTestData()
    #print(target_var)
    input_x  = input_x.cuda()
    target_y = target_var.cuda()
    ext=ExtractEmbedding(model,"fc")
    output=ext(input_x)

    xvector = feature.squeeze(0) # 2D to 1D
    # tensor to numpy (using detach to remove the variable grad)
    xvector = xvector.detach().numpy()
    # write xvector of utt
    kaldi_io.write_vec_flt(f, xvector, key=utt)

    #output dim 256 * 2048

def main():
    print("start time:",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) 
    model = load_model()
    ext = ExtractEmbedding(model,"layer4")
    dir="/home/work_nfs4_ssd/hzhao/feature/voxceleb1/test/feats.scp"
    dataset = getTestData(dir)
    batch_size=256
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    for i,trainset in enumerate(test_loader):
        (input_x,target_var) = trainset
        input_x=input_x.cuda()
        #print(input_x.size(),type(input_x))
        output=ext(input_x)
        output = output.squeeze()
        output = output.cpu()
        #tensor to numpy
        output = output.detach().numpy()
        #print(":::",len(target_var))
        target_var =np.squeeze(target_var)
        #tmp="/home/work_nfs/lizhang/node6/kaldi02/kaldi02/egs/voxceleb/v2/tmp"
        filename ="./test/99_test/enroll/xvector."+str(i)+".ark"
        


        f =kaldi_io.open_or_fd(filename,"wb")

        for i,uttid in enumerate(target_var):

            kaldi_io.write_vec_flt(f, output[i], key=uttid)
        
    print("end time:",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

if __name__ == '__main__':

    main()
