# !/usr/bin/python

# -*- coding: utf-8 -*-
import os
import sys
import argparse
import logging
from logging.handlers import RotatingFileHandler
import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("..")
import prepare.kaldi_io as kaldi_io
import pdb

def getArkList(xvectorscp):
    scpList =[]
    for line in open(xvectorscp,"r"):
        line = line.strip().split()[1].split(":")[0]
        if line not in scpList:
           scpList.append(line)
    return scpList

def cosine_enroll_mode(args, trials, enroll_embedding, eval_embedding):
    # load enroll spk-level embedding to enroll_dict
    enroll_dict = {}

    for spk, vec in kaldi_io.read_vec_flt_ark(enroll_embedding):
       # print(spk,vec) 
        vec = torch.from_numpy(vec) # 1D
        vec = vec.unsqueeze(0) # 1D to 2D
        enroll_dict[spk] = vec
    

    eval_dict = {}
    for line in open(eval_embedding,"r"):
        arkpath = line.split()[1]
        vec = kaldi_io.read_vec_flt(arkpath)
        vec = torch.from_numpy(vec)
        vec = vec.unsqueeze(0)
        uttid = line.split()[0]
     #   print(uttid) 
        eval_dict [uttid] = vec


    # load eval utt-level embedding to eval_dict
    '''
    for utt, vec in kaldi_io.read_vec_flt_ark(eval_embedding):
        vec = torch.from_numpy(vec) # 1D
        vec = vec.unsqueeze(0) # 1D to 2D
        eval_dict[utt] = vec
    '''
    #340 340
    ##print(len(enroll_dict),len(eval_dict))
    # compute cosine distance according to trials file
    # trials file 's format and scoring result file's format must be same
    f = open(args.result_file, 'w')

    cos = torch.nn.CosineSimilarity()

    with open(trials, 'r') as f_trials:

        for trial in f_trials:

            uttid,spk, _ = trial.strip().split()
            #spk,uttid, _ = trial.strip().split()
            dvector_enroll = enroll_dict[spk]
            #uttid=BAC009S0124W0127
            #uttid = uttid[6:]
            dvector_eval = eval_dict[uttid]
            #print(dvector_eval)
            score = round(cos(dvector_enroll, dvector_eval).item(), 6)

            f.write(uttid + ' ' + spk + ' ' + str(score) + '\n')

    f_trials.close()

    f.close()

def cosine_non_enroll_mode(args, trials, test_embedding):
    # load eval utt-level embedding to test_dict
    '''
    test_dict = {}
    for utt, vec in kaldi_io.read_vec_flt_ark(test_embedding):
        vec = torch.from_numpy(vec) # 1D
        vec = vec.unsqueeze(0) # 1D to 2D
        test_dict[utt] = vec
    '''
    test_dict = {}
    for line in open(test_embedding,"r"):
        arkpath = line.split()[1]


        if ".ark" in arkpath:
            vec = kaldi_io.read_vec_flt(arkpath)
            vec = torch.from_numpy(vec)
            vec = vec.unsqueeze(0)
            uttid = line.split()[0]
            #   print(uttid)
            #uttid = uttid[3:]
            test_dict [uttid] = vec
        elif ".npy" in arkpath:
                    
            vec = np.load(arkpath)
            
            vec = torch.from_numpy(vec)
            
            vec = vec.unsqueeze(0)
            
            uttid = line.split()[0]
            
            #   print(uttid) 
            #uttid = uttid[3:]
            test_dict [uttid] = vec

    # compute cosine distance according to trials file
    # trials file 's format and scoring result file's format must be same
    f = open(args.result_file, 'w')
    #cos = torch.nn.CosineSimilarity()
    with open(trials, 'r') as f_trials:
        for trial in f_trials:
            utt_1, utt_2,_ = trial.strip().split()
            #utt_1, utt_2,_  = trial.strip().split()
            '''
            utt_1 = utt_1[1:]
            utt_2 = utt_2[1:]
            '''
            #utt_1 = utt_1[6:]
            dvector_utt1 = test_dict[utt_1]
            dvector_utt2 = test_dict[utt_2]
            #score = round(cos(dvector_utt1, dvector_utt2).item(), 6)
            score = round(F.cosine_similarity(dvector_utt1, dvector_utt2).item(), 6)
            f.write(utt_1 + ' ' + utt_2 + ' ' + str(score) + '\n')
    f_trials.close()
    f.close()


def main(args):
    # load embedding file
    if args.enroll_mode:
        enroll_embedding = args.enroll_embedding
        eval_embedding = args.eval_embedding
    else:
        test_embedding = args.test_embedding

    # load trials file
    trials = args.trials

    # scoring
    if args.enroll_mode:
#
        #print("::::",trials, enroll_embedding, eval_embedding)
        cosine_enroll_mode(args, trials, enroll_embedding, eval_embedding)
    else:
        cosine_non_enroll_mode(args, trials, test_embedding)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute dvector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # embedding dir opts
    parser.add_argument('--enroll-mode', default=False,
                        action='store_true',
                        help="""If true, then input both enroll and eval embedding;
                        If false, just input test embedding(like voxceleb1_test).
                        Usage: python demo.py --enroll-mode(if you wana enroll_mode to be True) """)
    parser.add_argument('--enroll-embedding', type=str, default="test/aishell/xvector_25k/spk_xvector.ark",
                        help="Enroll embedding file. Used in enroll-mode")

    parser.add_argument('--eval-embedding', type=str, default="test/aishell/xvector_25k/xvector.scp",
                        help="Eval embedding file. Used in enroll-mode")
    #parser.add_argument('--test-embedding', type=str, default="/home/work_nfs3/lizhang/skyspace/validate/test/vox/dev/xvector.scp")
    parser.add_argument('--test-embedding', type=str, default="test/vox/29/xvector.scp")


    
    parser.add_argument('--trials', type=str, required=False, default="/home/work_nfs4_ssd/hzhao/feature/voxceleb1/test/trials",help="Trials file used to score. Note: Two formats[uttid uttid target/nontarget]/[uttid spk target/nontarget]according to non-enroll-mode/enroll-mode respectively")
    parser.add_argument('--result-file', type=str, required=False, default="arc_softmax",help="Scoring result file.Note: dir is same as embedding_dir/../")

    args = parser.parse_args()
    main(args)
