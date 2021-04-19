import pdb
import argparse
import yaml
import os
import shutil
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import sys
sys.path.append("..")
from torch.utils.data import DataLoader
from utils.parse_yaml import yaml_parse
from prepare.prepare import TorchDataset
from backbone.xvector import xvecTDNN 
from prepare.collate_fn import random_collate
from utils.logger import get_logger

from utils.utils import dropout_schedule,learning_rate_schedule, accuracy

def basicCofig(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = True
    logger = get_logger(verbosity=0, name='dataloader')
    return logger
    
def main(args):
    #global args, best_prec1
    global logger 
    logger = basicCofig(args)

    # prepare the model
    if args.arch == "xvecTDNN":
        #model = xvecTDNN(args.input_dim,int(args.numSpkrs), args.p_dropout)
        model = xvecTDNN()

    elif args.arch == "resnet":

        model = MainModel()

    #prepare optimizer
    if args.optimizer == "SGD":

       optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
       #optimizer = torch.optim.SGD( [model.parameters(),
                                   float(args.initial_lr),
                                   momentum=float(args.momentum),
                                   weight_decay=float(args.weight_decay))
    elif args['optimizer'] == 'Adam':
    
      optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=float(arg['initial_lr']), 
                                   betas=(0.9, 0.999),
                                   eps=1e-08, 
                                   weight_decay=float(args.weight_decay))
    

    model = torch.nn.DataParallel(model).cuda()
   
    logger.info(model)

    # get the number of model parameters
    logger.info('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            model.load_state_dict(torch.load(args.resume)['state_dict'],strict=True)
        else:
            logger.warning("=> no checkpoint found at '{}'".format(args.resume))

    #prepare validate_loader
    if args.validate == True:
        val_dataset =TorchDataset(args.validate_data,scp_type=args.scp_type)
        val_loader = DataLoader(
                   dataset=val_dataset,
                   batch_size=int(args.batch_size),
                   shuffle=True,
                   collate_fn=random_collate,
                   pin_memory=True,
                   drop_last=False)

        
    #preprare train_loader
    train_dataset = TorchDataset(args.train_data,scp_type=args.scp_type)
    train_loader = DataLoader(
                   dataset=train_dataset,
                   batch_size=args.batchsize,
                   shuffle=True,
                   collate_fn=random_collate,
                   pin_memory=True,
                   drop_last=False)

    # iteration training
    for epoch in range(args.start_epoch, args.epochs):
        

        lr_epoch = learning_rate_schedule(epoch,args.epochs,args.initial_lr,name="inverse_curve",final_lr=1e-5,step=5)
        dropout_epoch = dropout_schedule(args.dropout_schedule,epoch,args.epochs)

        #dynamically change dropout  & learning rate acordding epoch
        model.module.set_dropout(dropout_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_epoch

        
        # train for one epoch
        train(train_loader,model, optimizer, epoch,lr_epoch, dropout_epoch)
        
        # save each epoch model 
        best_prec1 = 0.0
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, args.model_prefix,epoch)
  

def train(train_loader, model, optimizer, epoch,lr_epoch,dropout_epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()
    
    for i, (input, target) in enumerate(train_loader):

        data_time.update(time.time() - end)
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target).cuda()
        embeddings,posterior, loss  = model(input_var,target_var)

        prec1, prec5 = accuracy(posterior.data, target_var, topk=(1, 5))

        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        optimizer.zero_grad()
 
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            logger.info(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'dropout {dropout:.4f} ({dropout:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, dropout = dropout_epoch,top1=top1, top5=top5))

def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        
        output = model(input_var)
        loss = criterion(output, target_var)
         
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
    
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    return top1.avg
    
def save_checkpoint(state, prefix,epoch):

    prefix = str(epoch)+"."+prefix

    filename=args.save_path+'/%s_checkpoint.pth.tar'%prefix
    
    torch.save(state, filename)
    logger.info("Finish saving epoch %s"%filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Speaker Vrification Models Training')

    parser.add_argument('--config', default="xvector.yml", type=str,help="yaml configure file")
    
    parsers = parser.parse_args()

    # loading parameters from configure file
    with open(parsers.config) as f:
         yaml_parameters = yaml.load(f,Loader=yaml.FullLoader)
    global args
    args = yaml_parse(yaml_parameters)
    main(args)

       

    
