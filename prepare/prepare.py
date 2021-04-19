import os
import pdb
import sys
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import torchvision.transforms as transforms
import random
import kaldi_python_io
from prepare.collate_fn import random_collate
#from collate_fn import random_collate

sys.path.append("..")
from utils.logger import get_logger

'''
If there are some bad cases, we can use TOLERANT_SAMPLE to avoid interruption of the whole programs.
'''
TOLERANT_LABEL ='448'
TOLERANT_SAMPLE ='/home/work_nfs3/lizhang/corpus/voxceleb2/one/exp/id00673/id0067300431-babble.npy'

logger = get_logger(verbosity=0, name='dataloader')

class TorchDataset(Dataset):
    '''
    This code is compatible with three types of scp files.
    feats.scp ...ark file
    feats.scp ...npy file
    wav.scp   ...wav file
    '''

    def __init__(self,data_path, scp_type='ark', resize_height=80, resize_width=256, repeat=1):
        self.utt2spk_file = os.path.join(data_path,"utt2spk")

        self.spk2utt_file = os.path.join(data_path,"spk2utt")
        
        assert os.path.exists(self.utt2spk_file)
        assert os.path.exists(self.spk2utt_file)

        if scp_type == 'ark':
           self.scp_file = os.path.join(data_path,"feats.scp")
           assert os.path.exists(self.scp_file)

           self.feats_reader = kaldi_python_io.ScriptReader(self.scp_file)   
 
           self.utt_list = self.feats_reader.index_keys

        elif scp_type == 'npy':
           pdb.set_trace()
           self.scp_file = os.path.join(data_path,"feats.scp")
           assert os.path.exists(self.scp_file)

           with open(self.scp_file) as  f:

                self.feats_reader = f.readlines()
        elif scp_type == 'wav':
           self.scp_file = os.path.join(data_path,"wav.scp")
           assert os.path.exists(self.scp_file)
 
        self.utt2spk, self.spk2utt = self.getMappingLabel(self.utt2spk_file, self.spk2utt_file) 
        self.scp_type = scp_type
        self.repeat = repeat 

        self.resize_height = resize_height

        self.resize_width = resize_width

    def getMappingLabel(self,utt2spk_file, spk2utt_file):
        utt2spk = {}
        spk2utt = {}
        with open(utt2spk_file) as f :
            for line in f:
                uttid = line.split()[0]
                spkid = line.split()[1]
                utt2spk[uttid] = spkid 

        index = -1
        with open(spk2utt_file) as f :
            for line in f:       
                index = index + 1
                spk2utt[line.split()[0]] = index

        return utt2spk, spk2utt

    def __getitem__(self, i):
        try :
            if self.scp_type == 'ark':
                uttid = self.feats_reader.index_keys[i]
                label = self.spk2utt[self.utt2spk[uttid]]
                return (self.feats_reader[uttid], label)

            elif self.scp_type == 'npy':
                uttid = self.feats_reader[i].split()[0]
            
                feats = np.load(self.feats_reader[i].split()[1]).transpose(1,0)
                label = self.spk2utt[self.utt2spk[uttid]]
                logger.error("Dataloader Loading ERROR ====> label:{} uttid:{}".format(uttid,label))
                return (feats, label)

            elif self.scp_type == 'wav':
                uttid = self.feats_reader[i].split()[0]
                wav = self.feats_reader[i].split()[1]
                label = self.spk2utt[self.utt2spk[uttid]]
                return (wav,label)
        except Exception as e:

               logger.error("Dataloader Loading ERROR ====> label:{} uttid:{}".format(uttid,label))
              
               print("error sample in dataloader loading:",label, uttid) 
               #load fault-tolerant sample
   
               return(np.load(TOLERANT_SAMPLE), TOLERANT_LABEL)

        '''
        try:
            mat = kaldi_io.read_mat(arkpath)
        except Exception as e:
            print("error---->",label,arkpath)
            path="/home/backup_nfs2/data-SPEAKER/lizhang/data/data/voxceleb2/mix_all/nosil/exp/xvector_feats_mix_all.56.ark:280170172" 
            mat = kaldi_io.read_mat(path)
            label = 3667

        orig_mat = mat
        mat_size = mat.shape

        while mat_size[0] < self.resize_width:
               
              mat = np.concatenate((mat,orig_mat),axis = 0)
              mat_size =mat.shape

        if mat_size[0] > self.resize_width:

           mat_ommit = mat_size[0] - self.resize_width

           tmp_start = np.random.randint(mat_ommit)

           tmp_end = tmp_start + self.resize_width

           mat = mat [tmp_start:tmp_end,:]

        audio = torch.from_numpy(mat).transpose(1,0).contiguous()
         
        return (audio, label)
        '''
    def __len__(self):
        if self.repeat == None:
            data_len = len(self.utt2spk.keys())
        else:
            data_len = len(self.utt2spk.keys()) * self.repeat
        return data_len
if __name__=='__main__':
    path="/home/work_nfs3/lizhang/corpus_feature/test_corpus/aishell_1/train/feats_number.scp"
    path="/home/work_nfs4_ssd/zhangli/data/voxceleb1/mfcc_80/"
    path="/home/work_nfs3/lizhang/corpus/voxceleb2/mix/"
    #path="/home/work_nfs4_ssd/zhangli/data/voxceleb1/train/"
    #path="/home/work_nfs3/lizhang/corpus_feature/voxceleb/mix_train_no_sil/number/train/feats_train_shuf.scp"
    dataset=TorchDataset(path,scp_type="npy")
    batch_size=5 
    train_loader = DataLoader(
                   dataset=dataset, 
                   batch_size=batch_size, 
                   shuffle=True,
                   collate_fn=random_collate,
                   pin_memory=True,
                   drop_last=False)
    print("end::",time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
    for i,trainset in enumerate(train_loader):

        features,targets = trainset
        print("features,targets::",targets,features,time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))  
        quit()
